import json
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from loswatch.history import (
    MEDICATIONS, SECTIONS, audit, filter_encounters, mapped_label,
    parse_mappings, read_encounters, stage_files,
)
from loswatch.ml.data import FEATURES, sha256


@pytest.fixture(autouse=True)
def isolated_streamlit_caches():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def history_files(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    mapping = directory / "IDS_mapping.csv"
    mapping.write_text(
        'admission_type_id,description\n1,Emergency\n6,NULL\n,\n'
        'discharge_disposition_id,description\n1,Discharged to home\n'
        '19,"Expired at home. Medicaid only, hospice."\n,\n'
        'admission_source_id,description\n1, Physician Referral\n7, Emergency Room\n17,NULL\n',
        encoding="utf-8",
    )
    rows = []
    for i in range(3):
        rows.append({
            "encounter_id": str(100 + i), "patient_nbr": "500" if i < 2 else "501",
            "age": "[50,60)", "gender": "Female", "time_in_hospital": i + 1,
            "admission_type_id": "1" if i < 2 else "99", "admission_source_id": "7",
            "discharge_disposition_id": "1", "number_outpatient": "0",
            "number_emergency": "0", "number_inpatient": "1",
            "diag_1": "250.83", "diag_2": "?", "diag_3": "V45",
            "readmitted": "NO", "change": "No", "diabetesMed": "Yes",
            "max_glu_serum": "None", "A1Cresult": "None",
            **{name: "Steady" if name == "insulin" else "No" for name in MEDICATIONS},
        })
    rows[1]["insulin"] = "No"
    rows[2]["diag_1"] = "401"
    csv_path = directory / "diabetic_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, mapping


def test_sections_separators_quoted_commas_and_null(history_files):
    mappings = parse_mappings(history_files[1])
    assert set(mappings) == set(SECTIONS)
    assert mappings["admission_type_id"]["1"] == "Emergency"
    assert mappings["discharge_disposition_id"]["1"] == "Discharged to home"
    assert mappings["admission_source_id"]["1"] == "Physician Referral"
    assert mappings["discharge_disposition_id"]["19"] == "Expired at home. Medicaid only, hospice."
    assert mappings["admission_type_id"]["6"] == "NULL"
    assert mapped_label(mappings["admission_type_id"], "99") == "Unmapped code 99"


@pytest.mark.parametrize("suffix", ["1,Duplicate\n", "unknown_section,description\n"])
def test_bad_mapping_rejected(history_files, suffix):
    mapping = history_files[1]
    mapping.write_text(mapping.read_text() + suffix)
    with pytest.raises(ValueError):
        parse_mappings(mapping)


def test_missing_and_literal_values(history_files):
    frame = read_encounters(history_files[0])
    assert frame.diag_2.isna().all()
    assert frame.diag_1.iloc[0] == "250.83"
    assert frame.max_glu_serum.eq("None").all()
    assert frame.metformin.eq("No").all()
    assert frame.time_in_hospital.tolist() == [1, 2, 3]


def test_search_literal_and_combined_filters(history_files):
    frame = read_encounters(history_files[0])
    assert len(filter_encounters(frame, patient="500")) == 2
    assert len(filter_encounters(frame, diagnosis="250.")) == 2
    assert len(filter_encounters(frame, diagnosis=".*")) == 0
    assert len(filter_encounters(frame, encounter="101", patient="500")) == 1
    assert len(filter_encounters(frame, patient="500", medication="insulin", status="Recorded use")) == 1
    assert len(filter_encounters(frame, medication="insulin", status="No")) == 1


def test_idempotent_staging_and_conflicts(history_files, tmp_path):
    destination = tmp_path / "raw"
    paths = stage_files(*history_files, directory=destination)
    before = [path.stat().st_mtime_ns for path in paths]
    assert stage_files(*history_files, directory=destination) == paths
    assert before == [path.stat().st_mtime_ns for path in paths]
    assert len(read_encounters(paths[0])) == 3
    paths[1].write_text("different")
    with pytest.raises(ValueError, match="Refusing"):
        stage_files(*history_files, directory=destination)


def test_duplicate_encounters_not_appended(history_files):
    csv_path = history_files[0]
    frame = pd.read_csv(csv_path, keep_default_na=False)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(csv_path, index=False)
    assert len(read_encounters(csv_path)) == 3
    frame.loc[1, "encounter_id"] = frame.loc[0, "encounter_id"]
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        read_encounters(csv_path)


def test_audit_identity_and_leakage_policy(history_files, tmp_path):
    phase2 = tmp_path / "phase2"
    phase2.mkdir()
    (phase2 / "metrics.json").write_text(json.dumps({"dataset_sha256": sha256(history_files[0]), "cleaning": {"clean_rows": 3}}))
    (phase2 / "feature_schema.json").write_text(json.dumps({"raw_required_features": FEATURES}))
    report = audit(*history_files, phase2_dir=phase2)
    assert report["phase2"]["same_csv_bytes"]
    assert report["phase2"]["schema_matches_allowlist"]
    assert not report["phase2"]["retrained"]
    assert report["missing_counts"]["diag_2"] == 3
    assert report["lab_not_measured_counts"]["A1Cresult"] == 3
    assert report["mappings"]["admission_type_id"]["unmapped_codes"] == {"99": 1}
    assert set(["discharge_disposition_id", "time_in_hospital", "readmitted", *MEDICATIONS]).issubset(report["ml_excluded_columns"])


def test_historical_fields_do_not_enter_model(history_files):
    pytest.importorskip("sklearn")
    from loswatch.ml.features import AdmissionFeatures
    frame = read_encounters(history_files[0])
    original = AdmissionFeatures().fit_transform(frame)
    for name in frame:
        if name not in FEATURES:
            frame[name] = "OUTCOME CHANGED"
    pd.testing.assert_frame_equal(original, AdmissionFeatures().fit_transform(frame))


def test_historical_ui_is_separate_and_searchable(history_files, factory, monkeypatch):
    from loswatch.models import Patient
    from loswatch.seed import seed_database
    from loswatch.services import records
    seed_database(factory)
    st.cache_resource.clear()
    monkeypatch.setenv("LOSWATCH_DATABASE_URL", str(factory.kw["bind"].url))
    monkeypatch.setenv("LOSWATCH_HISTORY_DIR", str(history_files[0].parent))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert app.metric[0].value == "20"
    app.sidebar.radio[0].set_value("Historical records").run()
    assert not app.exception
    assert app.metric[0].label == "Historical encounters"
    assert app.metric[0].value == "3"
    app.text_input[0].set_value("101").run()
    assert not app.exception
    assert app.selectbox[2].value == "101"
    assert app.metric[2].value == "2"
    assert not app.button
    app.text_input[0].set_value("not found").run()
    assert app.info
    app.sidebar.radio[0].set_value("Patients").run()
    assert app.metric[0].value == "20"
    with factory() as session:
        assert len(records(session, Patient)) == 20


def test_missing_historical_files_are_nonfatal(factory, tmp_path, monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setenv("LOSWATCH_DATABASE_URL", str(factory.kw["bind"].url))
    monkeypatch.setenv("LOSWATCH_HISTORY_DIR", str(tmp_path / "absent"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    app.sidebar.radio[0].set_value("Historical records").run()
    assert not app.exception
    assert "Historical data unavailable" in app.info[0].value
