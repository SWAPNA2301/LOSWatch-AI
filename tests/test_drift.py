import numpy as np
import pandas as pd
import pytest

from loswatch.drift import FEATURES, analyze, build_batches


def sample_frame(rows=200):
    values = {
        "age": ["[30-40)", "[50-60)", "[70-80)", "[40-50)"] * (rows // 4),
        "gender": ["Male", "Female"] * (rows // 2),
        "admission_type_id": ["1", "2", "3", "7"] * (rows // 4),
        "admission_source_id": ["1", "2", "7", "4"] * (rows // 4),
        "number_outpatient": np.tile([0, 1, 2, 3], rows // 4),
        "number_emergency": np.tile([0, 0, 1, 2], rows // 4),
        "number_inpatient": np.tile([0, 1, 1, 2], rows // 4),
    }
    return pd.DataFrame(values, columns=FEATURES)


def test_identical_batches_have_no_drift():
    frame = sample_frame()
    report = analyze(frame, frame, scenario="test-no-drift")
    assert report["drift_detected"] is False
    assert report["alerts"] == []
    assert all(result["psi"] == 0 for result in report["features"].values())


def test_numeric_shift_is_detected():
    reference = sample_frame()
    current = reference.copy()
    current["number_inpatient"] += 20
    report = analyze(reference, current, thresholds={"wasserstein": 2})
    assert report["features"]["number_inpatient"]["drift"]
    assert any(alert["feature"] == "number_inpatient" for alert in report["alerts"])


def test_missing_and_unseen_category_shifts_are_detected():
    reference = sample_frame()
    current = reference.copy()
    current.loc[:49, "number_outpatient"] = np.nan
    current.loc[:9, "admission_type_id"] = "999"
    report = analyze(reference, current)
    outpatient = report["features"]["number_outpatient"]
    admission_type = report["features"]["admission_type_id"]
    assert "missingness shift" in outpatient["alert_reasons"]
    assert admission_type["unseen_categories"] == ["999"]
    assert "unseen category" in admission_type["alert_reasons"]


def test_small_batches_are_safe_and_reproducible():
    reference = sample_frame(20)
    ref_a, current_a, label_a = build_batches(reference, size=2, seed=7, scenario="numeric_shift")
    ref_b, current_b, label_b = build_batches(reference, size=2, seed=7, scenario="numeric_shift")
    pd.testing.assert_frame_equal(ref_a, ref_b)
    pd.testing.assert_frame_equal(current_a, current_b)
    assert label_a == label_b
    report = analyze(ref_a.iloc[:1], current_a.iloc[:1])
    assert all(result["adjusted_pvalue"] is None for result in report["features"].values())


def test_invalid_batch_size_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        build_batches(sample_frame(), size=1)
