"""Offline synthetic tests: never download data or touch the demo database."""
import io
import json
from zipfile import ZipFile

import pytest

pytest.importorskip("sklearn", reason="Install requirements-ml.txt to run Phase 2 tests")
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from loswatch.ml.data import FEATURES, clean_rows, download
from loswatch.ml.features import AdmissionFeatures
from loswatch.ml.train import candidates, evaluate, make_pipeline, split_patients, train


@pytest.fixture
def sample():
    return pd.DataFrame({
        "encounter_id": range(120), "patient_nbr": np.repeat(range(60), 2),
        "time_in_hospital": np.tile(np.arange(1, 13), 10),
        "age": ["[50,60)"] * 120, "gender": ["Female", "Male"] * 60,
        "admission_type_id": [1] * 120, "admission_source_id": [7] * 120,
        "number_outpatient": np.tile([0, 1, 2], 40), "number_emergency": [0] * 120,
        "number_inpatient": [1] * 120, "readmitted": ["NO"] * 120,
        "discharge_disposition_id": [1] * 120, "num_medications": [10] * 120,
    })


def test_patient_splits_are_disjoint_and_repeatable(sample):
    splits = split_patients(sample)
    assert sum(len(part) for part in splits.values()) == len(sample)
    for name, part in splits.items():
        pd.testing.assert_frame_equal(part, split_patients(sample)[name])
        for other, other_part in splits.items():
            if name != other:
                assert not set(part.patient_nbr) & set(other_part.patient_nbr)


def test_outcomes_and_identifiers_cannot_change_features_or_predictions(sample):
    pipeline = make_pipeline(RandomForestRegressor(n_estimators=3, random_state=1)).fit(sample, sample.time_in_hospital)
    contaminated = sample.copy()
    for name in set(sample.columns) - set(FEATURES):
        contaminated[name] = 999999
    np.testing.assert_allclose(pipeline.predict(sample), pipeline.predict(contaminated))
    assert not set(AdmissionFeatures().fit_transform(sample).columns) & {"patient_nbr", "encounter_id", "time_in_hospital", "readmitted"}


def test_cleaning_and_conflicting_encounters(sample):
    dirty = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    dirty.loc[1, "time_in_hospital"] = 99
    dirty.loc[2, "patient_nbr"] = np.nan
    result, report = clean_rows(dirty)
    assert len(result) == 118
    assert report["exact_duplicates_removed"] == 1
    assert report["invalid_id_or_target_rows_removed"] == 2
    dirty = sample.copy()
    dirty.loc[1, "encounter_id"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        clean_rows(dirty)
    with pytest.raises(ValueError, match="Missing"):
        clean_rows(sample.drop(columns="age"))


def test_missing_unknown_and_engineering(sample):
    features = AdmissionFeatures().fit_transform(sample)
    assert features.age_midpoint.iloc[0] == 55
    assert features.prior_visits_total.iloc[2] == 3
    assert features.any_prior_inpatient.iloc[0] == 1
    broken = sample.astype({"number_inpatient": object}).copy()
    broken.loc[0, ["age", "gender", "number_inpatient"]] = ["?", "Unknown/Invalid", -1]
    broken.loc[0, "admission_type_id"] = 6
    transformed = AdmissionFeatures().fit_transform(broken)
    assert transformed.loc[0, ["age_midpoint", "gender", "number_inpatient", "prior_visits_total", "admission_type_id"]].isna().all()
    pipeline = make_pipeline(HistGradientBoostingRegressor(max_iter=2, early_stopping=False)).fit(sample, sample.time_in_hospital)
    broken.loc[0, "admission_source_id"] = 99
    assert np.isfinite(pipeline.predict(broken)).all()


def test_preprocessing_is_fitted_only_on_training_data(sample):
    pipeline = make_pipeline(DummyRegressor()).fit(sample, sample.time_in_hospital)
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    original = imputer.statistics_.copy()
    unseen = sample.copy()
    unseen["age"] = "[90,100)"
    pipeline.predict(unseen)
    np.testing.assert_equal(imputer.statistics_, original)
    assert original[0] == 55
    assert candidates()["HistGradientBoosting"].early_stopping is False


def test_metrics():
    result = evaluate([1, 2, 3], [2, 2, 2])
    assert result["MAE"] == pytest.approx(2 / 3)
    assert result["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert result["R2"] == 0


def test_artifacts_and_pipeline_roundtrip(sample, tmp_path):
    csv = tmp_path / "input.csv"
    sample.to_csv(csv, index=False)
    output = tmp_path / "run"
    metrics = train(csv, output, models={
        "DummyRegressor": DummyRegressor(strategy="median"),
        "RandomForest": RandomForestRegressor(n_estimators=3, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=2, early_stopping=False),
    })
    assert metrics["selected_model"] == min(metrics["models"], key=lambda name: (metrics["models"][name]["validation"]["MAE"], name))
    restored = joblib.load(output / "best_pipeline.joblib")
    assert np.isfinite(restored.predict(sample[FEATURES])).all()
    assert set(json.loads((output / "feature_schema.json").read_text())["raw_required_features"]) == set(FEATURES)
    assert json.loads((output / "eda.json").read_text())["encounters"] == metrics["split"]["train"]["rows"]
    assert len(pd.read_csv(output / "split_manifest.csv")) == len(sample)
    with pytest.raises(FileExistsError):
        train(csv, output)


def test_download_cache_provenance_and_no_arbitrary_extraction(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("diabetic_data.csv", "fake dataset")
        archive.writestr("IDS_mapping.csv", "fake mapping")
        archive.writestr("../escaped.txt", "untrusted path")
    monkeypatch.setattr("loswatch.ml.data.urlopen", lambda *args, **kwargs: io.BytesIO(payload.getvalue()))
    directory = tmp_path / "dataset"
    csv = download(directory)
    assert not (tmp_path / "escaped.txt").exists()
    monkeypatch.setattr("loswatch.ml.data.urlopen", lambda *args, **kwargs: pytest.fail("Cache should avoid network"))
    assert download(directory) == csv
    csv.write_text("tampered")
    with pytest.raises(ValueError, match="provenance"):
        download(directory)
