"""Deterministic, entirely fictional fixtures; never clinical recommendations."""
from datetime import date, timedelta

from sqlalchemy import func, select

from loswatch.database import initialize, make_engine
from loswatch.models import Admission, Bed, Diagnosis, Doctor, MedicationOrder, Patient

NAMES = (
    "Aarav Meadow", "Mira Brookfield", "Dev Rowan", "Anika Vale", "Rohan Ashford",
    "Leela Fern", "Ishan Willow", "Tara Westbrook", "Nikhil Cedar", "Sana Fairmont",
    "Arjun Glen", "Diya Wren", "Kabir Stonewell", "Nisha Hazel", "Vikram Finch",
    "Asha Linden", "Rehan Northwood", "Kavya Reed", "Neel Hartwell", "Riya Elm",
)


def seed_database(factory):
    """Seed only a completely empty database, atomically; repeated calls are no-ops."""
    with factory.begin() as session:
        if any(session.scalar(select(func.count()).select_from(model)) for model in (Patient, Doctor, Bed, Admission, Diagnosis, MedicationOrder)):
            return False
        doctors = [Doctor(name=name, specialty=specialty) for name, specialty in (
            ("Dr. Elara Moss (fictional)", "Internal Medicine"),
            ("Dr. Nolan Birch (fictional)", "Cardiology"),
            ("Dr. Priya Lake (fictional)", "Orthopedics"),
            ("Dr. Theo Arden (fictional)", "General Medicine"),
        )]
        beds = [Bed(label=f"{ward[0]}-{number:02}", ward=ward) for ward in ("General", "Observation", "Recovery") for number in range(1, 9)]
        session.add_all(doctors + beds)
        session.flush()
        diseases = ("Hypertension", "Type 2 diabetes", "Asthma", "Osteoarthritis", "Iron deficiency anemia")
        medications = ("Amlodipine", "Metformin", "Salbutamol", "Paracetamol", "Ferrous sulfate")
        today = date.today()
        for i, name in enumerate(NAMES):
            patient = Patient(mrn=f"DEMO-{i + 1:04}", name=name, date_of_birth=date(1955 + i * 2, i % 12 + 1, i % 27 + 1), sex="Female" if i % 2 else "Male", doctor_id=doctors[i % 4].id)
            session.add(patient)
            session.flush()
            admitted = today - timedelta(days=i % 9 + 2)
            session.add_all([
                Diagnosis(patient_id=patient.id, disease=diseases[i % 5], diagnosed_on=admitted - timedelta(days=120), notes="Entirely fictional demo diagnosis."),
                MedicationOrder(patient_id=patient.id, medication=medications[i % 5], instructions="Fictional recorded order; dose intentionally omitted. Not prescribing guidance.", start_on=admitted, end_on=None),
                Admission(patient_id=patient.id, bed_id=beds[i].id, admitted_on=admitted, discharged_on=None if i < 15 else today - timedelta(days=1), reason="Fictional scheduled assessment and observation"),
            ])
    return True


def main():
    engine = make_engine()
    try:
        print("Created 20 fictional patients." if seed_database(initialize(engine)) else "Database is not empty; seed skipped.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
