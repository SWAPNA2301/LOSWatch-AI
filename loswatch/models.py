"""Relational schema; deletion of referenced parents is deliberately restricted."""
from datetime import date

from sqlalchemy import CheckConstraint, Date, ForeignKey, Index, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Doctor(Base):
    __tablename__ = "doctors"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    specialty: Mapped[str] = mapped_column(String(120))
    __table_args__ = (CheckConstraint("length(trim(name)) > 0 AND length(trim(specialty)) > 0"),)


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    mrn: Mapped[str] = mapped_column(String(40), unique=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    date_of_birth: Mapped[date] = mapped_column(Date)
    sex: Mapped[str] = mapped_column(String(20))
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("doctors.id", ondelete="RESTRICT"))
    doctor: Mapped[Doctor | None] = relationship()
    diagnoses: Mapped[list["Diagnosis"]] = relationship(passive_deletes="all")
    medications: Mapped[list["MedicationOrder"]] = relationship(passive_deletes="all")
    admissions: Mapped[list["Admission"]] = relationship(passive_deletes="all")
    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0 AND length(trim(mrn)) > 0"),
        CheckConstraint("sex IN ('Female', 'Male', 'Other', 'Unknown')"),
    )


class Bed(Base):
    __tablename__ = "beds"
    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(40), unique=True)
    ward: Mapped[str] = mapped_column(String(120))
    __table_args__ = (CheckConstraint("length(trim(label)) > 0 AND length(trim(ward)) > 0"),)


class Admission(Base):
    __tablename__ = "admissions"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id", ondelete="RESTRICT"), index=True)
    bed_id: Mapped[int] = mapped_column(ForeignKey("beds.id", ondelete="RESTRICT"))
    admitted_on: Mapped[date] = mapped_column(Date)
    discharged_on: Mapped[date | None] = mapped_column(Date)
    reason: Mapped[str] = mapped_column(Text)
    bed: Mapped[Bed] = relationship()
    __table_args__ = (
        CheckConstraint("discharged_on IS NULL OR discharged_on >= admitted_on"),
        CheckConstraint("length(trim(reason)) > 0"),
        Index("uq_active_bed", "bed_id", unique=True, sqlite_where=text("discharged_on IS NULL")),
        Index("uq_active_patient", "patient_id", unique=True, sqlite_where=text("discharged_on IS NULL")),
    )


class Diagnosis(Base):
    __tablename__ = "diagnoses"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id", ondelete="RESTRICT"), index=True)
    disease: Mapped[str] = mapped_column(String(160))
    diagnosed_on: Mapped[date] = mapped_column(Date)
    notes: Mapped[str] = mapped_column(Text, default="")
    __table_args__ = (CheckConstraint("length(trim(disease)) > 0"),)


class MedicationOrder(Base):
    __tablename__ = "medication_orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id", ondelete="RESTRICT"), index=True)
    medication: Mapped[str] = mapped_column(String(160))
    instructions: Mapped[str] = mapped_column(Text)
    start_on: Mapped[date] = mapped_column(Date)
    end_on: Mapped[date | None] = mapped_column(Date)
    __table_args__ = (
        CheckConstraint("end_on IS NULL OR end_on >= start_on"),
        CheckConstraint("length(trim(medication)) > 0 AND length(trim(instructions)) > 0"),
    )
