"""Validated, atomic CRUD shared by the UI and seed command."""
from datetime import date

from sqlalchemy import Date, String, select
from sqlalchemy.exc import IntegrityError

from loswatch.models import Admission, Bed, Diagnosis, Doctor, MedicationOrder, Patient

MODELS = {model.__name__: model for model in (Patient, Doctor, Admission, Diagnosis, MedicationOrder, Bed)}


class ValidationError(ValueError):
    """An actionable user input or integrity error."""


def records(session, model):
    return list(session.scalars(select(model).order_by(model.id)))


def get_record(session, model, record_id):
    record = session.get(model, record_id)
    if record is None:
        raise ValidationError(f"{model.__name__} #{record_id} no longer exists. Refresh the page.")
    return record


def validate(session, model, values, record_id=None):
    columns = {column.name: column for column in model.__table__.columns if column.name != "id"}
    if set(values) != set(columns):
        raise ValidationError("Provide exactly the editable fields for this record.")
    cleaned = dict(values)
    for name, column in columns.items():
        value = cleaned[name]
        if value is None:
            if not column.nullable:
                raise ValidationError(f"{name} is required.")
            continue
        if isinstance(column.type, String):
            if not isinstance(value, str):
                raise ValidationError(f"{name} must be text.")
            value = value.strip()
            if not value and name != "notes":
                raise ValidationError(f"{name} cannot be blank.")
            if column.type.length and len(value) > column.type.length:
                raise ValidationError(f"{name} exceeds {column.type.length} characters.")
            cleaned[name] = value
        if isinstance(column.type, Date) and type(value) is not date:
            raise ValidationError(f"{name} must be a date.")
        for foreign_key in column.foreign_keys:
            target = next(m for m in MODELS.values() if m.__tablename__ == foreign_key.column.table.name)
            if type(value) is not int or session.get(target, value) is None:
                raise ValidationError(f"Select an existing {target.__name__.lower()}.")
    today = date.today()
    for name in ("date_of_birth", "admitted_on", "discharged_on", "diagnosed_on"):
        if cleaned.get(name) and cleaned[name] > today:
            raise ValidationError(f"{name} cannot be in the future.")
    if model is Patient:
        cleaned["mrn"] = cleaned["mrn"].upper()
        if cleaned["sex"] not in ("Female", "Male", "Other", "Unknown"):
            raise ValidationError("Select a valid sex value.")
        if record_id:
            for child, field in ((Admission, "admitted_on"), (Diagnosis, "diagnosed_on"), (MedicationOrder, "start_on")):
                if session.scalar(select(child.id).where(child.patient_id == record_id, getattr(child, field) < cleaned["date_of_birth"])):
                    raise ValidationError("Birth date cannot be later than existing patient records.")
    if model in (Admission, Diagnosis, MedicationOrder):
        patient = get_record(session, Patient, cleaned["patient_id"])
        field = {Admission: "admitted_on", Diagnosis: "diagnosed_on", MedicationOrder: "start_on"}[model]
        if cleaned[field] < patient.date_of_birth:
            raise ValidationError("Record date cannot precede the patient's birth date.")
    for start, end in (("admitted_on", "discharged_on"), ("start_on", "end_on")):
        if cleaned.get(end) and cleaned[end] < cleaned[start]:
            raise ValidationError(f"{end} cannot precede {start}.")
    return cleaned


def save_record(session, model, values, record_id=None):
    """Commit one complete record or roll back all changes on failure."""
    try:
        values = validate(session, model, values, record_id)
        record = get_record(session, model, record_id) if record_id is not None else model()
        for name, value in values.items():
            setattr(record, name, value)
        session.add(record)
        session.commit()
        return record
    except IntegrityError as exc:
        session.rollback()
        message = str(exc.orig)
        if "admissions.bed_id" in message:
            raise ValidationError("This bed is already occupied. Choose another bed or discharge its occupant.") from exc
        if "admissions.patient_id" in message:
            raise ValidationError("This patient already has an active admission.") from exc
        raise ValidationError("Duplicate identifier or invalid reference. MRNs and bed labels must be unique.") from exc
    except Exception:
        session.rollback()
        raise


def delete_record(session, model, record_id):
    try:
        session.delete(get_record(session, model, record_id))
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise ValidationError("This record is still referenced. Remove dependent records or reassign them first.") from exc
    except Exception:
        session.rollback()
        raise


def search_patients(session, query=""):
    query = query.strip()
    statement = select(Patient).order_by(Patient.name)
    if query:
        statement = statement.where(Patient.name.icontains(query, autoescape=True) | Patient.mrn.icontains(query, autoescape=True))
    return list(session.scalars(statement))


def active_admissions(session):
    return list(session.scalars(select(Admission).where(Admission.discharged_on.is_(None))))
