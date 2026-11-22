import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from loswatch.ml.data import FEATURES, TARGET
from loswatch.ml.phase21 import build_pipeline
from loswatch.ml.train import split_patients
from loswatch.retraining import evaluate_labeled_batch, evaluate_labeled_predictions, long_stay_metrics, performance_alert, promote_candidate, rollback_model, retraining_recommendation, train_candidates


def labeled_frame(rows=120):
    return pd.DataFrame({
        "encounter_id": np.arange(rows),
        "patient_nbr": np.repeat(np.arange(rows // 2), 2),
        "time_in_hospital": np.tile(np.arange(1, 13), rows // 12),
        "age": ["[50-60)"] * rows,
        "gender": ["Female", "Male"] * (rows // 2),
        "admission_type_id": [1] * rows,
        "admission_source_id": [7] * rows,
        "number_outpatient": np.tile([0, 1, 2], rows // 3),
        "number_emergency": [0] * rows,
        "number_inpatient": [1] * rows,
    })


def test_candidate_training_uses_patient_disjoint_validation(tmp_path):
    frame = labeled_frame()
    csv = tmp_path / "labeled.csv"
    frame.to_csv(csv, index=False)
    manifest = tmp_path / "split_manifest.csv"
    splits = split_patients(frame)
    pd.concat([part[["encounter_id", "patient_nbr"]].assign(split=name) for name, part in splits.items()]).to_csv(manifest, index=False)
    current = tmp_path / "current"
    current.mkdir()
    current_spec = {"name": "current", "family": "DummyRegressor", "engineered": False, "params": {"strategy": "median"}}
    joblib.dump(build_pipeline(current_spec).fit(frame[FEATURES], frame[TARGET]), current / "best_pipeline.joblib")
    output = tmp_path / "candidate"
    specs = [current_spec, {"name": "rf_small", "family": "RandomForest", "engineered": True, "params": {"n_estimators": 3, "max_depth": 3}}]
    result = train_candidates(csv, output, current, specs=specs, split_manifest=manifest)
    assert result["candidate_results"]
    actual_output = tmp_path / "candidate" / result["model_version"]
    metrics = json.loads((actual_output / "metrics.json").read_text())
    assert metrics["patient_overlap"] == 0
    assert metrics["split"]["train"]["patients"] + metrics["split"]["validation"]["patients"] + metrics["split"]["test"]["patients"] == 60
    assert actual_output == Path(result["output"]) and (actual_output / "best_pipeline.joblib").exists()


def test_candidate_runs_get_unique_directories_and_bad_labels_are_rejected(tmp_path):
    frame = labeled_frame()
    csv = tmp_path / "labeled.csv"
    frame.to_csv(csv, index=False)
    manifest = tmp_path / "split_manifest.csv"
    splits = split_patients(frame)
    pd.concat([part[["encounter_id", "patient_nbr"]].assign(split=name) for name, part in splits.items()]).to_csv(manifest, index=False)
    bad = frame.copy()
    bad.loc[0, TARGET] = 0
    bad.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="LOS"):
        train_candidates(csv, tmp_path / "candidates", tmp_path / "missing-current", specs=[], split_manifest=manifest)
    bad.to_csv(csv, index=False)
    current = tmp_path / "current"
    current.mkdir()
    spec = {"name": "current", "family": "DummyRegressor", "engineered": False, "params": {"strategy": "median"}}
    joblib.dump(build_pipeline(spec).fit(frame[FEATURES], frame[TARGET]), current / "best_pipeline.joblib")
    csv.write_text(frame.to_csv(index=False))
    first = train_candidates(csv, tmp_path / "candidates", current, specs=[spec], split_manifest=manifest)
    second = train_candidates(csv, tmp_path / "candidates", current, specs=[spec], split_manifest=manifest)
    assert first["output"] != second["output"]
    assert Path(first["output"]).exists() and Path(second["output"]).exists()


def test_original_holdout_identity_mismatch_is_rejected(tmp_path):
    frame = labeled_frame()
    manifest = tmp_path / "split_manifest.csv"
    splits = split_patients(frame)
    pd.concat([part[["encounter_id", "patient_nbr"]].assign(split=name) for name, part in splits.items()]).to_csv(manifest, index=False)
    corrupt = frame.copy()
    corrupt.loc[0, "patient_nbr"] = 999999
    csv = tmp_path / "corrupt.csv"
    corrupt.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="patient IDs"):
        train_candidates(csv, tmp_path / "candidate", tmp_path / "missing-current", specs=[], split_manifest=manifest)


def test_performance_metrics_include_long_stays():
    result = evaluate_labeled_predictions([1, 7, 10], [2, 8, 7])
    assert result["MAE"] == pytest.approx(5 / 3)
    assert result["long_stay"]["count"] == 2
    assert result["long_stay"]["MAE"] == pytest.approx(2.0)


def test_missing_labels_and_alert_recommendation():
    assert performance_alert(None)["available"] is False
    performance = {"MAE": 3.0, "RMSE": 3.8, "long_stay": {"MAE": 4.0}}
    result = retraining_recommendation({"drift_detected": True}, performance)
    assert result["recommend_retraining"]
    assert "data drift alert" in result["reasons"]
    assert "long-stay MAE threshold" in result["reasons"]


def test_simulated_drift_is_not_an_operational_retraining_trigger():
    result = retraining_recommendation({"drift_detected": True, "scenario": "Simulated drift: numeric shift", "simulated": True}, None)
    assert result["simulated_drift_ignored"]
    assert result["recommend_retraining"] is False


def test_full_original_csv_performance_uses_only_saved_test_holdout(tmp_path):
    frame = labeled_frame()
    manifest = tmp_path / "split_manifest.csv"
    splits = split_patients(frame)
    pd.concat([part[["encounter_id", "patient_nbr"]].assign(split=name) for name, part in splits.items()]).to_csv(manifest, index=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = {"name": "current", "family": "DummyRegressor", "engineered": False, "params": {"strategy": "median"}}
    joblib.dump(build_pipeline(spec).fit(frame[FEATURES], frame[TARGET]), model_dir / "best_pipeline.joblib")
    result = evaluate_labeled_batch(model_dir, frame, split_manifest=manifest)
    assert result["scope"] == "original_test_holdout"
    assert result["count"] == len(splits["test"])


def test_promotion_requires_approval_and_rollback_is_audited(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    pipeline = Pipeline([("model", DummyRegressor(strategy="mean"))]).fit([[1], [2]], [1, 2])
    joblib.dump(pipeline, candidate / "best_pipeline.joblib")
    (candidate / "feature_schema.json").write_text(json.dumps({"version": "2.1", "raw_required_features": FEATURES}))
    (candidate / "metrics.json").write_text(json.dumps({
        "selected_model": "RandomForest",
        "models": {"RandomForest": {"trial": "candidate-test", "validation": {"MAE": 1.0, "RMSE": 1.0, "long_stay": {"MAE": 1.0}}}},
        "current_model": {"validation": {"MAE": 2.0, "RMSE": 2.0, "long_stay": {"MAE": 2.0}}},
    }))
    (candidate / "candidate.json").write_text(json.dumps({"model_version": "candidate-test", "status": "candidate"}))
    registry = tmp_path / "registry"
    with pytest.raises(ValueError, match="approver"):
        promote_candidate(candidate, registry)
    destination = promote_candidate(candidate, registry, "reviewer", "validated")
    assert destination.exists()
    rollback_model("candidate-test", registry, "reviewer", "rollback test")
    events = [json.loads(line) for line in (registry / "audit.jsonl").read_text().splitlines()]
    assert [event["action"] for event in events] == ["promote", "rollback"]
    assert json.loads((registry / "current.json").read_text())["model_version"] == "candidate-test"


def test_weak_candidate_is_rejected_even_with_approval(tmp_path):
    candidate = tmp_path / "weak"
    candidate.mkdir()
    joblib.dump(Pipeline([("model", DummyRegressor(strategy="mean"))]).fit([[1], [2]], [1, 2]), candidate / "best_pipeline.joblib")
    (candidate / "feature_schema.json").write_text(json.dumps({"version": "2.1", "raw_required_features": FEATURES}))
    (candidate / "metrics.json").write_text(json.dumps({
        "selected_model": "RandomForest",
        "models": {"RandomForest": {"trial": "weak", "validation": {"MAE": 3.0, "RMSE": 3.0, "long_stay": {"MAE": 3.0}}}},
        "current_model": {"validation": {"MAE": 2.0, "RMSE": 2.0, "long_stay": {"MAE": 2.0}}},
    }))
    (candidate / "candidate.json").write_text(json.dumps({"model_version": "weak-test", "status": "candidate"}))
    with pytest.raises(ValueError, match="does not improve"):
        promote_candidate(candidate, tmp_path / "registry", "reviewer", "approved anyway")
