"""Synthetic regression tests for the versioned experiment and holdout boundary."""
import json

import pytest

pytest.importorskip("sklearn", reason="Install requirements-ml.txt for Phase 2.1")
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from loswatch.ml.data import FEATURES, sha256
from loswatch.ml.experiment21 import FrozenCorpus, error_analysis
from loswatch.ml.features import AdmissionFeatures
from loswatch.ml.features21 import AdmissionFeatures21
from loswatch.ml.phase21 import build_pipeline, develop, final_evaluate, trial_specs
from loswatch.ml.train import evaluate, split_patients


@pytest.fixture
def experiment(tmp_path):
    frame = pd.DataFrame({
        "encounter_id": np.arange(120), "patient_nbr": np.repeat(np.arange(60), 2),
        "time_in_hospital": np.tile(np.arange(1, 13), 10),
        "age": np.tile(["[0-10)", "[50-60)", "[90-100)"], 40),
        "gender": ["Female", "Male"] * 60,
        "admission_type_id": [1] * 120, "admission_source_id": [7] * 120,
        "number_outpatient": np.tile([0, 1, 2], 40), "number_inpatient": [1] * 120,
        "number_emergency": [0] * 120, "diag_1": ["250.83"] * 120,
        "discharge_disposition_id": [1] * 120, "readmitted": ["NO"] * 120,
        "insulin": ["Up"] * 120, "A1Cresult": ["None"] * 120,
    })
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    previous = tmp_path / "previous"
    previous.mkdir()
    splits = split_patients(frame)
    pd.concat([part[["encounter_id", "patient_nbr"]].assign(split=name) for name, part in splits.items()]).to_csv(previous / "split_manifest.csv", index=False)
    (previous / "metrics.json").write_text(json.dumps({"dataset_sha256": sha256(csv), "selected_model": "DummyRegressor", "models": {}}))
    specs = [
        {"name": "dummy", "family": "DummyRegressor", "engineered": False, "params": {"strategy": "median"}},
        {"name": "rf", "family": "RandomForest", "engineered": True, "params": {"n_estimators": 3, "max_depth": 3}},
        {"name": "hgb", "family": "HistGradientBoosting", "engineered": True, "params": {"max_iter": 3, "loss": "absolute_error", "min_samples_leaf": 5}},
    ]
    return frame, csv, previous, tmp_path / "output", specs


def test_age_regression_and_original_artifact_compatibility(experiment):
    frame = experiment[0]
    assert AdmissionFeatures().transform(frame).age_midpoint.isna().all()
    transformed = AdmissionFeatures21().fit_transform(frame)
    assert transformed.age_midpoint.iloc[:3].tolist() == [5, 55, 95]
    modified = frame.copy()
    modified.loc[0, "age"] = " [50,60) "
    modified.loc[1, "age"] = "[50-65)"
    modified.loc[2, "age"] = "[100-110)"
    result = AdmissionFeatures21().transform(modified)
    assert result.age_midpoint.iloc[0] == 55
    assert result.age_midpoint.iloc[1:3].isna().all()
    assert transformed.log1p_number_outpatient.iloc[2] == pytest.approx(np.log1p(2))
    assert transformed.prior_inpatient_share.iloc[2] == pytest.approx(1 / 3)
    assert transformed.admission_route.iloc[0] == "Emergency"


def test_all_nonadmission_fields_are_ignored(experiment):
    frame = experiment[0]
    before = AdmissionFeatures21().fit_transform(frame)
    for column in set(frame.columns) - set(FEATURES):
        frame[column] = "CHANGED OUTCOME"
    pd.testing.assert_frame_equal(before, AdmissionFeatures21().fit_transform(frame))
    with pytest.raises(ValueError):
        AdmissionFeatures21().transform(frame.drop(columns="age"))


def test_missing_values_unseen_categories_and_train_only_imputation(experiment):
    frame, _, _, _, specs = experiment
    pipeline = build_pipeline(specs[1]).fit(frame[FEATURES], frame.time_in_hospital)
    stats = pipeline.named_steps["preprocess"].named_transformers_["numeric"].statistics_.copy()
    changed = frame.astype({"number_inpatient": object}).copy()
    changed.loc[0, "age"] = "?"
    changed.loc[0, "number_inpatient"] = -1
    changed.loc[0, "admission_source_id"] = 999
    changed.loc[0, "gender"] = "Unknown/Invalid"
    assert np.isfinite(pipeline.predict(changed)).all()
    np.testing.assert_array_equal(stats, pipeline.named_steps["preprocess"].named_transformers_["numeric"].statistics_)
    assert stats[0] == 55


def test_split_audit_rejects_patient_overlap_and_identity_mismatch(experiment):
    frame, csv, previous, _, _ = experiment
    split_path = previous / "split_manifest.csv"
    original = pd.read_csv(split_path)
    corpus = FrozenCorpus(csv, split_path)
    loaded = corpus.load("train")
    assert set(loaded.encounter_id) == set(original.loc[original.split == "train", "encounter_id"])
    corrupt = original.copy()
    corrupt.loc[0, "split"] = "test" if corrupt.loc[0, "split"] != "test" else "train"
    corrupt.to_csv(split_path, index=False)
    with pytest.raises(ValueError, match="Patient overlap"):
        FrozenCorpus(csv, split_path)
    original.to_csv(split_path, index=False)
    frame.loc[0, "patient_nbr"] = 123456
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="exactly match"):
        FrozenCorpus(csv, split_path)


def test_development_never_loads_test_and_final_never_fits(experiment, monkeypatch):
    frame, csv, previous, output, specs = experiment
    original_load = FrozenCorpus.load
    accesses = []

    def development_load(self, name):
        assert name != "test", "Development must not load test outcomes/features"
        accesses.append(name)
        return original_load(self, name)

    monkeypatch.setattr(FrozenCorpus, "load", development_load)
    before = sha256(previous / "metrics.json")
    selected = develop(csv, previous, output, specs=specs)
    assert accesses == ["train", "validation"]
    assert sha256(previous / "metrics.json") == before
    assert not (output / "metrics.json").exists()
    assert not selected["test_evaluated"]
    audit = json.loads((output / "development_audit.json").read_text())
    assert audit["old_feature_missing_counts"]["age_midpoint"] == audit["split"]["train"]["rows"]
    assert audit["corrected_feature_missing_counts"]["age_midpoint"] == 0
    assert set(json.loads((output / "feature_schema.json").read_text())["raw_required_features"]) == set(FEATURES)

    def final_load(self, name):
        assert name == "test"
        accesses.append(name)
        return original_load(self, name)

    monkeypatch.setattr(FrozenCorpus, "load", final_load)
    monkeypatch.setattr(Pipeline, "fit", lambda *a, **k: pytest.fail("Final evaluation cannot fit"))
    metrics = final_evaluate(csv, previous, output)
    assert accesses == ["train", "validation", "test"]
    assert metrics["selected_model"] == selected["selected_model"]
    test = original_load(FrozenCorpus(csv, previous / "split_manifest.csv"), "test")
    restored = joblib.load(output / "best_pipeline.joblib")
    assert evaluate(test.time_in_hospital, restored.predict(test[FEATURES])) == metrics["models"][metrics["selected_model"]]["test"]
    with pytest.raises(FileExistsError):
        final_evaluate(csv, previous, output)
    with pytest.raises(FileExistsError):
        develop(csv, previous, output, specs=specs)


def test_final_refuses_modified_frozen_selection(experiment):
    _, csv, previous, output, specs = experiment
    develop(csv, previous, output, specs=specs)
    (output / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        final_evaluate(csv, previous, output)


def test_error_analysis_and_fixed_search(experiment):
    frame = experiment[0]
    report = error_analysis(frame, frame.time_in_hospital.to_numpy() + 1)
    assert report["mean_bias"] == 1
    assert report["patient_macro_MAE"] == 1
    assert sum(group["encounters"] for group in report["groups"]["los_band"]) == len(frame)
    specs = trial_specs()
    assert len(specs) == 11
    assert any(spec["params"].get("criterion") == "absolute_error" for spec in specs)
    for spec in specs:
        if spec["family"] == "HistGradientBoosting":
            assert build_pipeline(spec).named_steps["model"].early_stopping is False
