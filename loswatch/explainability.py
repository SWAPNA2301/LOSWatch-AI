"""Phase 6 SHAP explanations for the approved active LOS model."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from loswatch.ml.data import FEATURES, ROOT, sha256
from loswatch.ml.experiment21 import FrozenCorpus
from loswatch.prediction import ModelBundle, PredictionError, active_model_dir, artifact_signature, load_bundle

MAX_EXPLANATION_ROWS = 200
MAX_BACKGROUND_ROWS = 100
REPORT_DIR = ROOT / "artifacts" / "phase6" / "reports"
REFERENCE_DATA = ROOT / "data" / "raw" / "diabetic_data.csv"
ORIGINAL_SPLIT_MANIFEST = ROOT / "artifacts" / "phase2" / "split_manifest.csv"
LIMITATIONS = [
    "SHAP values are associative explanations of this model, not causal effects.",
    "The LOS prediction is experimental and is not clinical advice or a discharge recommendation.",
    "Only admission-time inputs are used; explanations do not establish what would happen under an intervention.",
]


class ExplanationError(ValueError):
    """A safe, actionable explanation error for the UI."""


_NUMERIC_LABELS = {
    "age_midpoint": "Age at admission",
    "number_outpatient": "Outpatient visits (prior year)",
    "number_emergency": "Emergency visits (prior year)",
    "number_inpatient": "Inpatient visits (prior year)",
    "prior_visits_total": "Total prior-year visits",
    "any_prior_inpatient": "Any prior inpatient visit",
    "log1p_number_outpatient": "Log outpatient visits (prior year)",
    "log1p_number_emergency": "Log emergency visits (prior year)",
    "log1p_number_inpatient": "Log inpatient visits (prior year)",
    "log1p_prior_visits_total": "Log total prior-year visits",
    "prior_inpatient_share": "Prior inpatient visit share",
    "prior_emergency_share": "Prior emergency visit share",
    "any_prior_visits": "Any prior-year visits",
    "age_x_prior_inpatient": "Age x prior inpatient indicator",
}


def readable_feature_label(feature_name: str) -> str:
    """Map a transformed model feature to a stable, user-facing label.

    The raw transformed name is retained in reports so this presentation label
    cannot hide or change the model's exact feature mapping.
    """
    name = str(feature_name)
    raw = name.split("__", 1)[-1]
    missing_indicator = raw.startswith("missingindicator_")
    if missing_indicator:
        raw = raw.removeprefix("missingindicator_")

    label = _NUMERIC_LABELS.get(raw)
    if label is None:
        match = re.fullmatch(r"gender_(.+)", raw)
        if match:
            label = f"Gender: {match.group(1)}"
        match = re.fullmatch(r"admission_type_id_(.+)", raw)
        if match:
            from loswatch.prediction import ADMISSION_TYPES
            code = match.group(1)
            label = f"Admission type: {ADMISSION_TYPES.get(int(code), code) if code.isdigit() else code}"
        match = re.fullmatch(r"admission_source_id_(.+)", raw)
        if match:
            from loswatch.prediction import ADMISSION_SOURCES
            code = match.group(1)
            label = f"Admission source: {ADMISSION_SOURCES.get(int(code), code) if code.isdigit() else code}"
        match = re.fullmatch(r"admission_route_(.+)", raw)
        if match:
            label = f"Admission route: {match.group(1)}"
        match = re.fullmatch(r"age_band_(\d+)", raw)
        if match:
            start = int(match.group(1)) * 10
            label = f"Age band [{start}, {start + 10})"
    if label is None:
        label = raw.replace("_", " ").strip().capitalize()
    return f"Missing: {label}" if missing_indicator else label


def load_validation_sample(
    csv_path: str | Path = REFERENCE_DATA,
    split_manifest: str | Path = ORIGINAL_SPLIT_MANIFEST,
    *,
    max_rows: int = MAX_BACKGROUND_ROWS,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one deterministic validation encounter per patient.

    ``FrozenCorpus`` validates the complete saved identity map and split
    manifest before loading the validation rows. This makes the sample
    patient-disjoint from train/test and prevents the dashboard from silently
    substituting the original test holdout.
    """
    if max_rows < 1 or max_rows > MAX_EXPLANATION_ROWS:
        raise ExplanationError(f"Validation sample size must be between 1 and {MAX_EXPLANATION_ROWS}.")
    try:
        corpus = FrozenCorpus(csv_path, split_manifest)
        validation = corpus.load("validation")
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        raise ExplanationError(f"Validation explanation sample unavailable: {exc}") from exc
    if validation.empty:
        raise ExplanationError("The saved validation partition is empty; global importance cannot be computed.")
    representatives = (
        validation.sort_values(["patient_nbr", "encounter_id"], kind="mergesort")
        .drop_duplicates("patient_nbr", keep="first")
    )
    if len(representatives) > max_rows:
        representatives = representatives.sample(n=max_rows, random_state=seed)
    representatives = representatives.sort_values("encounter_id", kind="mergesort").reset_index(drop=True)
    if representatives["patient_nbr"].duplicated().any():
        raise ExplanationError("The validation explanation sample contains duplicate patients.")
    metadata = {
        "partition": "validation",
        "sample_size": int(len(representatives)),
        "patient_count": int(representatives["patient_nbr"].nunique()),
        "patient_disjoint": True,
        "manifest_sha256": sha256(split_manifest),
        "source_sha256": corpus.dataset_hash,
        "source_path": str(csv_path),
        "seed": seed,
    }
    return representatives.loc[:, FEATURES].copy(), metadata


def combine_explanation_reports(
    patient_report: dict[str, Any],
    global_report: dict[str, Any],
    validation_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Keep patient SHAP results while attaching validation-sample importance."""
    if patient_report.get("model_version") != global_report.get("model_version"):
        raise ExplanationError("Patient and global explanations were generated by different model versions.")
    report = dict(patient_report)
    report["global_importance"] = global_report["global_importance"]
    report["global_sample_size"] = int(global_report["sample_size"])
    report["global_patient_count"] = int(validation_metadata["patient_count"])
    report["global_partition"] = validation_metadata["partition"]
    report["global_patient_disjoint"] = bool(validation_metadata["patient_disjoint"])
    report["global_sample_source"] = validation_metadata["source_path"]
    report["global_manifest_sha256"] = validation_metadata["manifest_sha256"]
    report["global_source_sha256"] = validation_metadata["source_sha256"]
    report["global_max_reconstruction_error"] = global_report["max_reconstruction_error"]
    return report


def _import_shap():
    try:
        import shap
        return shap
    except ImportError as exc:
        raise ExplanationError("Explainability unavailable: install the optional SHAP dependency from requirements-ml.txt.") from exc


def _bundle_model_version(bundle: ModelBundle) -> str:
    return str(bundle.metrics.get("model_version") or bundle.metrics.get("selected_trial") or "unknown")


def _validate_frame(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ExplanationError(f"Explanation input is missing admission-time features: {sorted(missing)}")
    if len(frame) == 0:
        raise ExplanationError("At least one admission is required for explanation.")
    if len(frame) > cap:
        raise ExplanationError(f"Explanation batch has {len(frame)} rows; the safe limit is {cap}.")
    return frame.loc[:, FEATURES].copy()


def _feature_names(bundle: ModelBundle) -> list[str]:
    names = bundle.pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    if names != bundle.schema.get("transformed_features"):
        raise ExplanationError("Explanation feature names do not match the approved model schema.")
    return names


def _shap_values(shap, model, transformed: np.ndarray):
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed, check_additivity=False)
        expected = explainer.expected_value
    except Exception as exc:
        raise ExplanationError(f"Explainability unavailable for unsupported active model type: {type(model).__name__}.") from exc
    if isinstance(values, list):
        if len(values) != 1:
            raise ExplanationError("Only single-output LOS regression models are supported.")
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ExplanationError("SHAP returned an unsupported contribution shape.")
    expected = float(np.asarray(expected).reshape(-1)[0])
    return values, expected


def explain_batch(bundle: ModelBundle, frame: pd.DataFrame, *, background: pd.DataFrame | None = None, max_rows: int = MAX_EXPLANATION_ROWS, max_background: int = MAX_BACKGROUND_ROWS, tolerance: float = 1e-5) -> dict[str, Any]:
    """Explain a bounded raw-input batch, including the saved preprocessing pipeline."""
    if max_rows < 1 or max_rows > MAX_EXPLANATION_ROWS:
        raise ExplanationError(f"max_rows must be between 1 and {MAX_EXPLANATION_ROWS}.")
    raw = _validate_frame(frame, max_rows)
    if background is not None:
        _validate_frame(background, min(max_background, MAX_BACKGROUND_ROWS))
    shap = _import_shap()
    pipeline = bundle.pipeline
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "predict"):
        raise ExplanationError("Explainability unavailable: the active model is not a supported regressor.")
    try:
        transformed = pipeline[:-1].transform(raw)
        predictions = np.asarray(model.predict(transformed), dtype=float)
    except Exception as exc:
        raise ExplanationError("Explainability unavailable: approved model preprocessing could not transform the admission inputs.") from exc
    names = _feature_names(bundle)
    if transformed.shape[1] != len(names):
        raise ExplanationError("Transformed feature count does not match the approved schema.")
    values, base_value = _shap_values(shap, model, transformed)
    if values.shape != (len(raw), len(names)):
        raise ExplanationError("SHAP contribution shape does not match the transformed feature schema.")
    reconstruction = base_value + values.sum(axis=1)
    max_error = float(np.max(np.abs(reconstruction - predictions)))
    if max_error > tolerance:
        raise ExplanationError(f"SHAP contributions do not reconstruct model predictions within tolerance; maximum error was {max_error:.6g}.")
    patient_rows = []
    for row_index, (prediction, contribution_row) in enumerate(zip(predictions, values)):
        contributions = {name: float(value) for name, value in zip(names, contribution_row) if abs(value) > 0}
        labeled = [{"feature": name, "label": readable_feature_label(name), "contribution": value} for name, value in contributions.items()]
        patient_rows.append({"row": row_index, "prediction_days": float(prediction), "base_value": base_value, "contributions": contributions, "labeled_contributions": labeled})
    importance = pd.DataFrame({"feature": names, "label": [readable_feature_label(name) for name in names], "mean_abs_shap": np.mean(np.abs(values), axis=0), "mean_shap": np.mean(values, axis=0)}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return {"schema_version": "phase6.v1", "model_version": _bundle_model_version(bundle), "model_dir": str(active_model_dir()), "created_utc": datetime.now(timezone.utc).isoformat(), "sample_size": len(raw), "background_size": min(len(background), max_background) if background is not None else 0, "raw_features": FEATURES, "transformed_features": names, "feature_labels": {name: readable_feature_label(name) for name in names}, "base_value": base_value, "predictions_days": predictions.tolist(), "max_reconstruction_error": max_error, "patient_contributions": patient_rows, "global_importance": importance.to_dict(orient="records"), "limitations": LIMITATIONS}


def save_report(report: dict[str, Any], directory: str | Path = REPORT_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    version = str(report.get("model_version", "unknown")).replace("/", "-")
    path = directory / f"explanation_{stamp}_{version}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path


def load_active_bundle():
    directory = active_model_dir()
    try:
        return load_bundle(directory), directory, artifact_signature(directory)
    except PredictionError as exc:
        raise ExplanationError(str(exc)) from exc
