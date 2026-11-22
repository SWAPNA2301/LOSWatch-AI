"""Phase 5: offline performance monitoring, retraining and model lifecycle."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits

from loswatch.ml.data import FEATURES, HISTORY, ROOT, TARGET, clean_rows, sha256
from loswatch.ml.phase21 import build_pipeline, trial_specs
from loswatch.ml.train import evaluate, split_patients, write_json

PHASE = "5"
DEFAULT_DATA = ROOT / "data" / "raw" / "diabetic_data.csv"
DEFAULT_CURRENT = ROOT / "artifacts" / "phase21"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase5" / "candidates"
DEFAULT_SPLIT_MANIFEST = ROOT / "artifacts" / "phase2" / "split_manifest.csv"
REGISTRY_DIR = ROOT / "artifacts" / "model_registry"
LONG_STAY_DAYS = 7


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def long_stay_metrics(actual, predicted, threshold=LONG_STAY_DAYS) -> dict[str, Any]:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = actual >= threshold
    if not mask.any():
        return {"threshold_days": threshold, "count": 0, "MAE": None, "RMSE": None, "mean_error": None}
    errors = predicted[mask] - actual[mask]
    return {"threshold_days": threshold, "count": int(mask.sum()), "MAE": float(mean_absolute_error(actual[mask], predicted[mask])), "RMSE": float(np.sqrt(mean_squared_error(actual[mask], predicted[mask]))), "mean_error": float(errors.mean())}


def evaluate_labeled_predictions(actual, predicted, long_stay_threshold=LONG_STAY_DAYS) -> dict[str, Any]:
    """Return aggregate and long-stay metrics for a labeled batch."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Actual and predicted LOS arrays must be nonempty and have equal length.")
    result = evaluate(actual, predicted)
    result["count"] = int(len(actual))
    result["long_stay"] = long_stay_metrics(actual, predicted, long_stay_threshold)
    return result


def evaluate_labeled_batch(model_dir: str | Path, frame: pd.DataFrame, long_stay_threshold=LONG_STAY_DAYS, split_manifest: str | Path = DEFAULT_SPLIT_MANIFEST) -> dict[str, Any] | None:
    """Evaluate only when actual LOS labels are present; never fit a model."""
    if TARGET not in frame.columns:
        return None
    clean, _ = clean_rows(frame)
    clean, scope = _monitoring_scope(clean, split_manifest)
    pipeline = joblib.load(Path(model_dir) / "best_pipeline.joblib")
    predicted = pipeline.predict(clean[FEATURES])
    result = evaluate_labeled_predictions(clean[TARGET], predicted, long_stay_threshold)
    result["model_dir"] = str(model_dir)
    result["dataset_sha256"] = sha256(Path(model_dir) / "best_pipeline.joblib")
    result["scope"] = scope
    return result


def _validate_retraining_frame(raw: pd.DataFrame) -> None:
    """Reject malformed labeled data before it can enter a training split."""
    required = {"encounter_id", "patient_nbr", TARGET, *FEATURES}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Retraining data is missing required columns: {sorted(missing)}")
    for name in ("encounter_id", "patient_nbr"):
        values = pd.to_numeric(raw[name], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"Retraining data contains invalid {name} values.")
    if raw["encounter_id"].duplicated().any():
        raise ValueError("Retraining data contains duplicate encounter_id values.")
    target = pd.to_numeric(raw[TARGET], errors="coerce")
    if target.isna().any() or (~target.between(1, 14) | (target % 1 != 0)).any():
        raise ValueError("Retraining labels must be integer LOS values from 1 to 14 days.")
    for name in HISTORY:
        values = pd.to_numeric(raw[name].replace({"": np.nan, "?": np.nan}), errors="coerce")
        invalid = raw[name].notna() & raw[name].astype(str).str.strip().ne("") & values.isna()
        if invalid.any() or (values.dropna() < 0).any() or (values.dropna() % 1 != 0).any():
            raise ValueError(f"Retraining feature {name} contains invalid nonnegative integer counts.")
    age = raw["age"].replace({"": np.nan, "?": np.nan}).dropna().astype(str).str.strip()
    if not age.str.match(r"^\[\d+\s*[-,]\s*\d+\)$").all():
        raise ValueError("Retraining feature age contains an invalid age band.")


def _load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Original split manifest not found: {path}")
    manifest = pd.read_csv(path)
    required = {"encounter_id", "patient_nbr", "split"}
    if not required.issubset(manifest.columns) or not bool(manifest["split"].isin({"train", "validation", "test"}).all()):
        raise ValueError("Original split manifest has invalid columns or split names.")
    if manifest["encounter_id"].duplicated().any():
        raise ValueError("Original split manifest contains duplicate encounter IDs.")
    groups = [set(manifest.loc[manifest.split == name, "patient_nbr"]) for name in ("train", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Original split manifest has patient overlap.")
    return manifest


def _preserved_splits(frame: pd.DataFrame, manifest_path: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Reuse original assignments and reject unknown or mismatched identities."""
    manifest = _load_manifest(manifest_path)
    indexed = manifest.set_index("encounter_id")
    ids = set(frame["encounter_id"])
    unknown = ids - set(indexed.index)
    if unknown:
        raise ValueError("Retraining data contains encounters absent from the original split manifest; refusing to guess a holdout assignment.")
    expected_patients = frame["encounter_id"].map(indexed["patient_nbr"])
    if not np.array_equal(frame["patient_nbr"].to_numpy(), expected_patients.to_numpy()):
        raise ValueError("Retraining data patient IDs do not match the original split manifest.")
    splits = {name: frame.loc[frame["encounter_id"].isin(manifest.loc[manifest.split == name, "encounter_id"])].copy() for name in ("train", "validation", "test")}
    if splits["train"].empty or splits["validation"].empty:
        raise ValueError("Retraining data must include original train and validation patients.")
    return splits, {"manifest_sha256": sha256(manifest_path), "preserved_test_rows": len(splits["test"]), "preserved_test_patients": int(splits["test"].patient_nbr.nunique()), "validation_partition_sha256": _partition_hash(splits["validation"])}


def _partition_hash(frame: pd.DataFrame) -> str:
    values = frame[["encounter_id", "patient_nbr"]].sort_values(["encounter_id", "patient_nbr"]).to_csv(index=False).encode()
    import hashlib
    return hashlib.sha256(values).hexdigest()


def _monitoring_scope(frame: pd.DataFrame, manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST) -> tuple[pd.DataFrame, str]:
    """When a full historical CSV is supplied, score only its untouched test holdout."""
    if not Path(manifest_path).exists() or "encounter_id" not in frame:
        return frame, "unpartitioned_labeled_batch"
    manifest = _load_manifest(manifest_path)
    indexed = manifest.set_index("encounter_id")
    overlap = set(frame["encounter_id"]) & set(indexed.index)
    if not overlap:
        return frame, "unpartitioned_labeled_batch"
    if set(frame["encounter_id"]) - set(indexed.index):
        raise ValueError("Monitoring data mixes original encounters with unknown encounters; provide one coherent batch.")
    expected = frame["encounter_id"].map(indexed["patient_nbr"])
    if not np.array_equal(frame["patient_nbr"].to_numpy(), expected.to_numpy()):
        raise ValueError("Monitoring patient IDs do not match the original split manifest.")
    test_ids = set(manifest.loc[manifest.split == "test", "encounter_id"])
    holdout = frame.loc[frame["encounter_id"].isin(test_ids)].copy()
    if holdout.empty:
        raise ValueError("Monitoring batch contains no original test-holdout encounters.")
    return holdout, "original_test_holdout"


def performance_alert(performance: dict[str, Any] | None, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thresholds = {"MAE": 2.5, "RMSE": 3.5, "long_stay_MAE": 3.0, **(thresholds or {})}
    if performance is None:
        return {"available": False, "alert": False, "reasons": [], "thresholds": thresholds}
    reasons = []
    if performance["MAE"] > thresholds["MAE"]:
        reasons.append("overall MAE threshold")
    if performance["RMSE"] > thresholds["RMSE"]:
        reasons.append("overall RMSE threshold")
    long_stay = performance["long_stay"]
    if long_stay["MAE"] is not None and long_stay["MAE"] > thresholds["long_stay_MAE"]:
        reasons.append("long-stay MAE threshold")
    return {"available": True, "alert": bool(reasons), "reasons": reasons, "thresholds": thresholds}


def retraining_recommendation(drift_report: dict[str, Any] | None, performance: dict[str, Any] | None, performance_thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    performance_result = performance_alert(performance, performance_thresholds)
    simulated = bool(drift_report and (drift_report.get("simulated") or str(drift_report.get("scenario", "")).startswith("Simulated drift")))
    drift_alert = bool(drift_report and drift_report.get("drift_detected") and not simulated)
    reasons = []
    if drift_alert:
        reasons.append("data drift alert")
    reasons.extend(performance_result["reasons"])
    return {"recommend_retraining": bool(reasons), "reasons": reasons, "drift_alert": drift_alert, "simulated_drift_ignored": simulated, "performance_alert": performance_result["alert"], "performance_available": performance_result["available"]}


def _candidate_version() -> str:
    return "candidate-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]


def train_candidates(csv_path=DEFAULT_DATA, output=DEFAULT_OUTPUT, current_model_dir=DEFAULT_CURRENT, seed=42, specs=None, long_stay_threshold=LONG_STAY_DAYS, split_manifest=DEFAULT_SPLIT_MANIFEST) -> dict[str, Any]:
    """Train candidates on labeled eligible data and compare on one patient-disjoint validation set."""
    raw = pd.read_csv(csv_path, low_memory=False, keep_default_na=False, na_values=["?"])
    _validate_retraining_frame(raw)
    frame, cleaning = clean_rows(raw)
    splits, split_audit = _preserved_splits(frame, split_manifest)
    specs = [spec for spec in (trial_specs() if specs is None else specs) if spec["family"] in {"DummyRegressor", "RandomForest", "HistGradientBoosting"}]
    if not specs:
        raise ValueError("At least one supported candidate specification is required.")
    validation = splits["validation"]
    current_pipeline = joblib.load(Path(current_model_dir) / "best_pipeline.joblib")
    current_metrics = evaluate_labeled_predictions(validation[TARGET], current_pipeline.predict(validation[FEATURES]), long_stay_threshold)
    results = []
    best = None
    version = _candidate_version()
    output = Path(output) / version
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=2):
        for spec in specs:
            started = time.monotonic()
            pipeline = build_pipeline(spec, seed).fit(splits["train"][FEATURES], splits["train"][TARGET])
            metrics = evaluate_labeled_predictions(validation[TARGET], pipeline.predict(validation[FEATURES]), long_stay_threshold)
            result = {"name": spec["name"], "family": spec["family"], "spec": spec, "validation": metrics, "elapsed_seconds": round(time.monotonic() - started, 3)}
            results.append(result)
            if best is None or (metrics["MAE"], metrics["RMSE"], spec["name"]) < (best["validation"]["MAE"], best["validation"]["RMSE"], best["name"]):
                best = result
                joblib.dump(pipeline, output / "best_pipeline.joblib")
    best_pipeline = joblib.load(output / "best_pipeline.joblib")
    model_type = best["family"]
    engineered_features = best_pipeline.named_steps["admission_features"].get_feature_names_out().tolist()
    transformed_features = best_pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    write_json(output / "feature_schema.json", {"version": "2.1", "raw_required_features": FEATURES, "selected_engineered": bool(best["spec"].get("engineered")), "engineered_features": engineered_features, "transformed_features": transformed_features, "source": "Phase 5 candidate; same seven admission-time inputs as Phase 2.1"})
    metrics = {"phase": PHASE, "selected_model": model_type, "selected_trial": best["name"], "model_version": version, "dataset_sha256": sha256(csv_path), "split": {name: {"rows": len(part), "patients": int(part.patient_nbr.nunique())} for name, part in splits.items()}, "patient_overlap": 0, "selection_rule": "Lowest identical-validation MAE, then RMSE, then name; no test set used", "models": {model_type: {"trial": best["name"], "validation": best["validation"]}}, "candidate_results": results, "current_model": {"validation": current_metrics, "model_dir": str(current_model_dir), "validation_partition_sha256": split_audit["validation_partition_sha256"]}, "split_audit": split_audit, "created_utc": _utc(), "versions": {"sklearn": __import__("sklearn").__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    write_json(output / "metrics.json", metrics)
    write_json(output / "candidate.json", {"schema_version": "phase5.v1", "model_version": version, "status": "candidate", "model_type": model_type, "created_utc": metrics["created_utc"], "dataset_sha256": metrics["dataset_sha256"]})
    pd.concat([part[["encounter_id", "patient_nbr"]].assign(split=name) for name, part in splits.items()]).to_csv(output / "split_manifest.csv", index=False)
    write_json(output / "audit.json", {"action": "candidate_trained", "timestamp_utc": metrics["created_utc"], "model_version": version, "eligible_rows": len(frame), "cleaning": cleaning, "patient_disjoint": True, "split_audit": split_audit, "test_holdout_used_for_training": False})
    write_json(output / "artifact_manifest.json", {name: sha256(output / name) for name in ("best_pipeline.joblib", "feature_schema.json", "metrics.json")})
    return {"output": str(output), "model_version": version, "selected_model": model_type, "selected_trial": best["name"], "candidate": best["validation"], "current": current_metrics, "candidate_results": results}


def _audit(registry: Path, action: str, version: str, approved_by: str, details: dict[str, Any] | None = None):
    registry.mkdir(parents=True, exist_ok=True)
    with (registry / "audit.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"action": action, "timestamp_utc": _utc(), "model_version": version, "approved_by": approved_by, "details": details or {}}, allow_nan=False) + "\n")


def _candidate_improves_active(metrics: dict[str, Any]) -> tuple[bool, str]:
    selected = metrics.get("selected_model")
    candidate = metrics.get("models", {}).get(selected, {}).get("validation")
    active = metrics.get("current_model", {}).get("validation")
    if not candidate or not active:
        return False, "Candidate is missing the active-model comparison required for promotion."
    candidate_key = (candidate.get("MAE", float("inf")), candidate.get("RMSE", float("inf")))
    active_key = (active.get("MAE", float("inf")), active.get("RMSE", float("inf")))
    if candidate_key >= active_key:
        return False, "Candidate does not improve active-model validation MAE/RMSE."
    candidate_long = candidate.get("long_stay", {}).get("MAE")
    active_long = active.get("long_stay", {}).get("MAE")
    if candidate_long is not None and active_long is not None and candidate_long > active_long:
        return False, "Candidate worsens active-model long-stay MAE."
    return True, "Candidate improves active-model validation metrics."


def promote_candidate(candidate_dir: str | Path, registry_dir: str | Path = REGISTRY_DIR, approved_by: str = "", reason: str = "") -> Path:
    if not approved_by.strip():
        raise ValueError("Explicit approver identity is required for promotion.")
    candidate_dir, registry_dir = Path(candidate_dir), Path(registry_dir)
    metadata = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    metrics = json.loads((candidate_dir / "metrics.json").read_text(encoding="utf-8"))
    improves, message = _candidate_improves_active(metrics)
    if not improves:
        raise ValueError(message)
    version = metadata["model_version"]
    destination = registry_dir / version
    if destination.exists():
        raise FileExistsError("This candidate version is already registered.")
    destination.mkdir(parents=True)
    for name in ("best_pipeline.joblib", "feature_schema.json", "metrics.json", "candidate.json"):
        shutil.copy2(candidate_dir / name, destination / name)
    manifest_source = candidate_dir / "artifact_manifest.json"
    if manifest_source.exists():
        shutil.copy2(manifest_source, destination / "artifact_manifest.json")
    else:
        (destination / "artifact_manifest.json").write_text(json.dumps({name: sha256(destination / name) for name in ("best_pipeline.joblib", "feature_schema.json", "metrics.json")}, indent=2), encoding="utf-8")
    (destination / "candidate.json").write_text(json.dumps({**metadata, "status": "promoted", "promoted_utc": _utc(), "approved_by": approved_by, "approval_reason": reason}, indent=2), encoding="utf-8")
    (registry_dir / "current.json").write_text(json.dumps({"model_version": version, "path": str(destination), "approved_by": approved_by, "promoted_utc": _utc()}, indent=2), encoding="utf-8")
    _audit(registry_dir, "promote", version, approved_by, {"reason": reason})
    return destination


def rollback_model(version: str, registry_dir: str | Path = REGISTRY_DIR, approved_by: str = "", reason: str = "") -> Path:
    if not approved_by.strip():
        raise ValueError("Explicit approver identity is required for rollback.")
    registry_dir = Path(registry_dir)
    destination = registry_dir / version
    if not (destination / "best_pipeline.joblib").exists():
        raise FileNotFoundError(f"Registered model version not found: {version}")
    (registry_dir / "current.json").write_text(json.dumps({"model_version": version, "path": str(destination), "approved_by": approved_by, "rolled_back_utc": _utc()}, indent=2), encoding="utf-8")
    _audit(registry_dir, "rollback", version, approved_by, {"reason": reason})
    return destination


def active_model_dir(registry_dir: str | Path = REGISTRY_DIR, fallback: str | Path = DEFAULT_CURRENT) -> Path:
    registry_dir = Path(registry_dir)
    pointer = registry_dir / "current.json"
    if pointer.exists():
        path = Path(json.loads(pointer.read_text(encoding="utf-8"))["path"])
        if (path / "best_pipeline.joblib").exists():
            return path
    return Path(fallback)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train_candidates(args.csv, args.output, args.current, args.seed), indent=2))


if __name__ == "__main__":
    main()
