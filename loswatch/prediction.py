"""Validated inference from trusted local Phase 2.1 artifacts. Never fits models."""
from dataclasses import dataclass
from datetime import date, timedelta
import json
import logging
import math
from pathlib import Path
import re
import warnings

import pandas as pd

from loswatch.ml.data import FEATURES, HISTORY, ROOT, sha256

MODEL_DIR = ROOT / "artifacts" / "phase21"
ARTIFACTS = ("best_pipeline.joblib", "feature_schema.json", "metrics.json")
LOGGER = logging.getLogger(__name__)
AGES = tuple(f"[{start}-{start + 10})" for start in range(0, 100, 10))
# UCI IDS_mapping.csv labels, excluding documented missing/unavailable codes.
ADMISSION_TYPES = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 7: "Trauma Center"}
ADMISSION_SOURCES = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 10: "Transfer from critical access hospital",
    11: "Normal Delivery", 12: "Premature Delivery", 13: "Sick Baby", 14: "Extramural Birth",
    18: "Transfer From Another Home Health Agency", 19: "Readmission to Same Home Health Agency",
    22: "Transfer from hospital inpatient/same facility resulting in a separate claim",
    23: "Born inside this hospital", 24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
}


class PredictionError(ValueError):
    """An actionable input or artifact problem safe to display in the UI."""


@dataclass(frozen=True)
class ModelBundle:
    pipeline: object
    schema: dict
    metrics: dict


@dataclass(frozen=True)
class LOSPrediction:
    total_days: float
    estimated_discharge: date | None


def artifact_signature(directory=MODEL_DIR):
    """Content-based cache key; missing files never silently use a cached model."""
    directory = Path(directory)
    try:
        return tuple((name, sha256(directory / name)) for name in (*ARTIFACTS, "artifact_manifest.json"))
    except OSError as exc:
        raise PredictionError("Prediction unavailable: the local Phase 2.1 model, schema, metrics or checksum manifest is missing or unreadable.") from exc


def load_bundle(directory=MODEL_DIR):
    """Only operator-controlled local artifacts; never accept a joblib upload."""
    directory = Path(directory)
    try:
        signature = dict(artifact_signature(directory))
        manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
        if any(manifest.get(name) != signature[name] for name in ARTIFACTS):
            raise PredictionError("Prediction unavailable: Phase 2.1 artifact checksums do not match. Restore the original files.")
        schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if schema.get("version") != "2.1" or schema.get("raw_required_features") != FEATURES:
            raise PredictionError("Prediction unavailable: incompatible admission feature schema.")
        if metrics.get("phase") != "2.1" or metrics.get("selected_model") != "RandomForest":
            raise PredictionError("Prediction unavailable: these artifacts are not the selected Phase 2.1 Random Forest.")
        model_metrics = metrics["models"]["RandomForest"]
        if model_metrics["trial"] != metrics["selected_trial"]:
            raise PredictionError("Prediction unavailable: model and evaluation metadata disagree.")
        for name in ("MAE", "RMSE", "R2"):
            value = model_metrics["test"][name]
            if type(value) not in (float, int) or not math.isfinite(value) or (name != "R2" and value < 0):
                raise PredictionError("Prediction unavailable: invalid model evaluation metrics.")
        import joblib
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.exceptions import InconsistentVersionWarning
        from sklearn.pipeline import Pipeline
        from sklearn.utils.validation import check_is_fitted
        from loswatch.ml.features21 import AdmissionFeatures21

        if metrics["versions"]["sklearn"] != sklearn.__version__:
            raise PredictionError(f"Prediction unavailable: install scikit-learn {metrics['versions']['sklearn']} to match the saved model.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = joblib.load(directory / "best_pipeline.joblib")
        if not isinstance(pipeline, Pipeline) or not isinstance(pipeline.named_steps.get("model"), RandomForestRegressor):
            raise PredictionError("Prediction unavailable: expected a Random Forest pipeline.")
        transformer = pipeline.named_steps.get("admission_features")
        if not isinstance(transformer, AdmissionFeatures21):
            raise PredictionError("Prediction unavailable: expected the Phase 2.1 admission transformer.")
        check_is_fitted(pipeline.named_steps["model"])
        if transformer.get_feature_names_out().tolist() != schema["engineered_features"]:
            raise PredictionError("Prediction unavailable: engineered feature schema does not match the model.")
        if pipeline.named_steps["preprocess"].get_feature_names_out().tolist() != schema["transformed_features"]:
            raise PredictionError("Prediction unavailable: transformed feature schema does not match the model.")
        return ModelBundle(pipeline, schema, metrics)
    except PredictionError:
        raise
    except ImportError as exc:
        raise PredictionError("Prediction unavailable: install the optional dependencies from requirements-ml.txt.") from exc
    except Exception as exc:
        LOGGER.exception("Failed to load local prediction artifacts")
        raise PredictionError("Prediction unavailable: the saved artifacts could not be loaded. Check their format and dependency versions.") from exc


def validate_inputs(values, schema):
    """Require known inputs. Missing never becomes zero or an invented patient fact."""
    if schema.get("raw_required_features") != FEATURES or set(values) != set(FEATURES):
        raise PredictionError("Provide exactly the seven admission-time inputs. Outcome fields are not accepted.")
    cleaned = {}
    missing = []
    for name in FEATURES:
        value = values[name]
        if value is None or (isinstance(value, str) and value.strip() in ("", "?", "Unknown", "Unknown/Invalid")) or (isinstance(value, float) and math.isnan(value)):
            missing.append(name.replace("_", " "))
        else:
            cleaned[name] = value.strip() if isinstance(value, str) else value
    if missing:
        raise PredictionError("Prediction requires known values for: " + ", ".join(missing) + ". No values have been substituted.")
    if cleaned["age"] not in AGES:
        raise PredictionError("Select the patient's known age band at admission (0–99 years).")
    if cleaned["gender"] not in ("Male", "Female"):
        raise PredictionError("This model requires a known supported gender category; it must not be inferred.")
    for name in [*HISTORY, "admission_type_id", "admission_source_id"]:
        value = cleaned[name]
        if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
            raise PredictionError(f"{name.replace('_', ' ')} must be a nonnegative whole number.")
        if len(str(value)) > 15:
            raise PredictionError(f"{name.replace('_', ' ')} is too large.")
        cleaned[name] = int(value)
    for name, allowed in (("admission_type_id", ADMISSION_TYPES), ("admission_source_id", ADMISSION_SOURCES)):
        if cleaned[name] not in allowed:
            raise PredictionError(f"Select a known {name.replace('_id', '').replace('_', ' ')}; unavailable codes cannot be used.")
        trained_name = f"categorical__{name}_{cleaned[name]}"
        if trained_name not in schema["transformed_features"]:
            raise PredictionError(f"This {name.replace('_id', '').replace('_', ' ')} was absent from training; no prediction is available for it.")
    return cleaned


def validate_admission_date(admitted_on):
    if admitted_on is not None and (type(admitted_on) is not date or admitted_on > date.today()):
        raise PredictionError("Use a known admission date on or before today, or leave it blank.")


def predict_los(bundle, values, admitted_on=None):
    cleaned = validate_inputs(values, bundle.schema)
    validate_admission_date(admitted_on)
    try:
        predictions = bundle.pipeline.predict(pd.DataFrame([cleaned], columns=FEATURES))
        if len(predictions) != 1:
            raise ValueError("Unexpected prediction shape")
        total_days = float(predictions[0])
        if not math.isfinite(total_days) or not 1 <= total_days <= 14:
            raise ValueError("Invalid LOS prediction")
        # A date-only admission cannot support a time-of-day estimate. Round up
        # the total duration solely for the explicitly uncertain calendar date.
        discharge = admitted_on + timedelta(days=math.ceil(total_days)) if admitted_on else None
        return LOSPrediction(total_days, discharge)
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise PredictionError("Prediction could not be calculated. Check the inputs and local model files.") from exc
