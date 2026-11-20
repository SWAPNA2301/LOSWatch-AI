from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from threading import Barrier

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from loswatch.models import Admission, Bed, Diagnosis, Doctor, MedicationOrder, Patient
from loswatch.seed import seed_database
from loswatch.services import ValidationError, active_admissions, delete_record, records, save_record, search_patients


def values(record):
    return {c.name: getattr(record, c.name) for c in record.__table__.columns if c.name != "id"}


@pytest.fixture
def seeded(factory):
    seed_database(factory)
    return factory


def test_seed_is_complete_and_idempotent(factory):
    assert seed_database(factory)
    assert not seed_database(factory)
    with factory() as session:
        assert [session.scalar(select(func.count()).select_from(m)) for m in (Patient, Doctor, Bed, Admission, Diagnosis, MedicationOrder)] == [20, 4, 24, 20, 20, 20]
        assert len(active_admissions(session)) == 15
        patient = session.get(Patient, 1)
        assert patient.doctor and patient.diagnoses and patient.medications and patient.admissions


def test_search_is_case_insensitive_and_literal(seeded):
    with seeded() as session:
        assert len(search_patients(session, "demo-0001")) == 1
        assert len(search_patients(session, "  aarav  ")) == 1
        assert search_patients(session, "%") == []
        assert search_patients(session, "'") == []


def test_bed_transfer_discharge_and_reuse(seeded):
    with seeded() as session:
        admission = session.get(Admission, 1)
        data = values(admission)
        with pytest.raises(ValidationError, match="occupied"):
            save_record(session, Admission, dict(data, bed_id=2), 1)
        assert session.get(Admission, 1).bed_id == 1
        save_record(session, Admission, dict(data, bed_id=24), 1)
        save_record(session, Admission, dict(data, bed_id=24, discharged_on=date.today()), 1)
        save_record(session, Admission, dict(data, bed_id=24, admitted_on=date.today()))
        assert len(active_admissions(session)) == 15


def test_duplicate_active_patient_and_reopen(seeded):
    with seeded() as session:
        data = values(session.get(Admission, 1))
        with pytest.raises(ValidationError, match="active admission"):
            save_record(session, Admission, dict(data, bed_id=24))
        old = values(session.get(Admission, 16))
        with pytest.raises(ValidationError, match="occupied"):
            save_record(session, Admission, dict(old, discharged_on=None, bed_id=1), 16)


@pytest.mark.parametrize("model,record_id,changes", [
    (Patient, 1, {"name": "  "}),
    (Patient, 1, {"date_of_birth": date.today() + timedelta(days=1)}),
    (Patient, 1, {"date_of_birth": date.today()}),
    (Patient, 1, {"sex": "invalid"}),
    (Patient, 1, {"doctor_id": 999}),
    (Patient, 1, {"mrn": "demo-0002"}),
    (Doctor, 1, {"name": "x" * 121}),
    (Bed, 1, {"label": "G-02"}),
    (Admission, 1, {"discharged_on": date(1900, 1, 1)}),
    (Admission, 1, {"admitted_on": date.today() + timedelta(days=1)}),
    (Diagnosis, 1, {"diagnosed_on": date(1900, 1, 1)}),
    (Diagnosis, 1, {"patient_id": 999}),
    (MedicationOrder, 1, {"end_on": date(1900, 1, 1)}),
    (MedicationOrder, 1, {"instructions": ""}),
])
def test_invalid_updates_rollback(seeded, model, record_id, changes):
    with seeded() as session:
        original = values(session.get(model, record_id))
        with pytest.raises(ValidationError):
            save_record(session, model, dict(original, **changes), record_id)
        assert values(session.get(model, record_id)) == original


@pytest.mark.parametrize("model", [Patient, Doctor, Bed])
def test_referenced_parents_cannot_be_deleted(seeded, model):
    with seeded() as session:
        with pytest.raises(ValidationError, match="referenced"):
            delete_record(session, model, 1)
        assert session.get(model, 1)


def test_crud_all_models_and_assignment(factory):
    with factory() as session:
        doctor = save_record(session, Doctor, {"name": "Dr. Fiction", "specialty": "General"})
        bed = save_record(session, Bed, {"label": "A1", "ward": "General"})
        patient = save_record(session, Patient, {"mrn": "test-1", "name": "Fiction Person", "date_of_birth": date(1990, 1, 1), "sex": "Unknown", "doctor_id": doctor.id})
        admission = save_record(session, Admission, {"patient_id": patient.id, "bed_id": bed.id, "admitted_on": date.today(), "discharged_on": None, "reason": "Demo"})
        diagnosis = save_record(session, Diagnosis, {"patient_id": patient.id, "disease": "Demo condition", "diagnosed_on": date.today(), "notes": ""})
        medication = save_record(session, MedicationOrder, {"patient_id": patient.id, "medication": "Demo", "instructions": "Fictional", "start_on": date.today(), "end_on": None})
        for record in (doctor, bed, patient, admission, diagnosis, medication):
            save_record(session, type(record), values(record), record.id)
        save_record(session, Patient, dict(values(patient), doctor_id=None), patient.id)
        assert session.get(Patient, patient.id).doctor_id is None
        for record in (medication, diagnosis, admission, patient, doctor, bed):
            delete_record(session, type(record), record.id)
            assert not records(session, type(record))


def test_database_constraints_bypass_service(seeded):
    with seeded() as session:
        session.add(Admission(patient_id=16, bed_id=1, admitted_on=date.today(), reason="Demo"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.add(Diagnosis(patient_id=999, disease="Demo", diagnosed_on=date.today(), notes=""))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.add(Admission(patient_id=16, bed_id=24, admitted_on=date.today(), discharged_on=date(1900, 1, 1), reason="Demo"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_concurrent_bed_claim_has_one_winner(seeded):
    barrier = Barrier(2)

    def claim(patient_id):
        with seeded() as session:
            barrier.wait(timeout=10)
            try:
                save_record(session, Admission, {"patient_id": patient_id, "bed_id": 24, "admitted_on": date.today(), "discharged_on": None, "reason": "Demo"})
                return "saved"
            except ValidationError as exc:
                assert "occupied" in str(exc)
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, [16, 17])) == ["conflict", "saved"]


def test_missing_and_unknown_fields(factory):
    with factory() as session:
        with pytest.raises(ValidationError):
            save_record(session, Doctor, {"name": "Name"})
        with pytest.raises(ValidationError):
            delete_record(session, Doctor, 999)
