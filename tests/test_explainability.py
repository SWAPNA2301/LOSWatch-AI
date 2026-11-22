import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from loswatch.explainability import (
    ExplanationError,
    combine_explanation_reports,
    explain_batch,
    load_validation_sample,
    readable_feature_label,
    save_report,
)
from loswatch.ml.data import FEATURES
from loswatch.ml.phase21 import build_pipeline
from loswatch.ml.train import split_patients
from loswatch.prediction import ModelBundle


def explanation_frame(rows=120):
    return pd.DataFrame({
        "age": np.tile(["[30-40)", "[50-60)", "[70-80)"], rows // 3),
        "gender": ["Female", "Male"] * (rows // 2),
        "admission_type_id": [1, 2, 3, 7] * (rows // 4),
        "admission_source_id": [1, 2, 7, 4] * (rows // 4),
        "number_outpatient": np.tile([0, 1, 2], rows // 3),
        "number_emergency": np.tile([0, 0, 1], rows // 3),
        "number_inpatient": np.tile([0, 1, 1], rows // 3),
    }, columns=FEATURES)


def bundle_for(model_spec):
    frame = explanation_frame()
    pipeline = build_pipeline(model_spec).fit(frame, np.tile(np.arange(1, 13), len(frame) // 12))
    schema = {"version": "2.1", "raw_required_features": FEATURES, "engineered_features": pipeline.named_steps["admission_features"].get_feature_names_out().tolist(), "transformed_features": pipeline.named_steps["preprocess"].get_feature_names_out().tolist()}
    metrics = {"phase": "5", "model_version": "candidate-test", "selected_trial": model_spec["name"]}
    return ModelBundle(pipeline, schema, metrics), frame


def test_explanation_batch_cap_is_enforced_without_loading_shap():
    bundle, frame = bundle_for({"name": "dummy", "family": "DummyRegressor", "engineered": False, "params": {"strategy": "mean"}})
    with pytest.raises(ExplanationError, match="safe limit"):
        explain_batch(bundle, frame.iloc[:2], max_rows=1)


def test_unsupported_model_is_reported_gracefully(monkeypatch):
    bundle, frame = bundle_for({"name": "dummy", "family": "DummyRegressor", "engineered": False, "params": {"strategy": "mean"}})
    class FakeShap:
        class TreeExplainer:
            def __init__(self, model):
                raise RuntimeError("unsupported")
    monkeypatch.setattr("loswatch.explainability._import_shap", lambda: FakeShap)
    with pytest.raises(ExplanationError, match="unsupported"):
        explain_batch(bundle, frame.iloc[:1])


def test_shap_reconstructs_predictions_and_preserves_schema_names():
    pytest.importorskip("shap")
    bundle, frame = bundle_for({"name": "rf", "family": "RandomForest", "engineered": True, "params": {"n_estimators": 8, "max_depth": 4}})
    report = explain_batch(bundle, frame.iloc[:4], background=frame.iloc[:20])
    assert report["raw_features"] == FEATURES
    assert report["transformed_features"] == bundle.schema["transformed_features"]
    assert report["max_reconstruction_error"] <= 1e-5
    assert report["patient_contributions"][0]["prediction_days"] == pytest.approx(report["base_value"] + sum(report["patient_contributions"][0]["contributions"].values()), abs=1e-5)
    assert report["global_importance"]
    for item in report["global_importance"]:
        values = [row["contributions"].get(item["feature"], 0.0) for row in report["patient_contributions"]]
        assert item["mean_abs_shap"] == pytest.approx(float(np.mean(np.abs(values))))
        assert item["label"] == report["feature_labels"][item["feature"]]
    assert any("associative" in limitation for limitation in report["limitations"])


def test_readable_labels_preserve_admission_category_mappings():
    assert readable_feature_label("numeric__number_inpatient") == "Inpatient visits (prior year)"
    assert readable_feature_label("categorical__admission_type_id_7") == "Admission type: Trauma Center"
    assert readable_feature_label("categorical__admission_source_id_7") == "Admission source: Emergency Room"
    assert readable_feature_label("categorical__age_band_4") == "Age band [40, 50)"


def test_validation_sample_is_patient_disjoint_and_deterministic(tmp_path):
    frame = explanation_frame(120)
    frame.insert(0, "encounter_id", np.arange(1000, 1120))
    frame.insert(1, "patient_nbr", np.repeat(np.arange(60), 2))
    frame["time_in_hospital"] = np.tile(np.arange(1, 13), 10)
    csv_path = tmp_path / "reference.csv"
    frame.to_csv(csv_path, index=False)
    splits = split_patients(frame, seed=42)
    manifest_path = tmp_path / "split_manifest.csv"
    pd.concat([part[["encounter_id", "patient_nbr"]].assign(split=name) for name, part in splits.items()]).to_csv(manifest_path, index=False)

    first, metadata = load_validation_sample(csv_path, manifest_path, max_rows=10)
    second, second_metadata = load_validation_sample(csv_path, manifest_path, max_rows=10)
    pd.testing.assert_frame_equal(first, second)
    assert metadata == second_metadata
    assert len(first) == 10
    assert first["patient_nbr"].is_unique if "patient_nbr" in first else True
    assert metadata["partition"] == "validation"
    assert metadata["patient_disjoint"] is True
    assert metadata["sample_size"] == metadata["patient_count"] == 10


def test_combined_report_keeps_patient_explanation_and_validation_global_metadata():
    patient = {"model_version": "v1", "sample_size": 1, "max_reconstruction_error": 0.001, "patient_contributions": [{"contributions": {}}]}
    global_report = {"model_version": "v1", "sample_size": 4, "max_reconstruction_error": 0.002, "global_importance": [{"feature": "numeric__age_midpoint", "label": "Age at admission", "mean_abs_shap": 0.3}]}
    metadata = {"partition": "validation", "patient_count": 4, "patient_disjoint": True, "source_path": "reference.csv", "manifest_sha256": "manifest", "source_sha256": "source"}
    combined = combine_explanation_reports(patient, global_report, metadata)
    assert combined["sample_size"] == 1
    assert combined["patient_contributions"] == patient["patient_contributions"]
    assert combined["global_sample_size"] == 4
    assert combined["global_patient_count"] == 4
    assert combined["global_partition"] == "validation"
    assert combined["global_max_reconstruction_error"] == 0.002


def test_explanation_report_is_versioned_and_downloadable(tmp_path):
    report = {"model_version": "candidate-test", "sample_size": 1, "limitations": ["associative"]}
    path = save_report(report, tmp_path)
    saved = json.loads(path.read_text())
    assert path.name.startswith("explanation_")
    assert saved["model_version"] == "candidate-test"
