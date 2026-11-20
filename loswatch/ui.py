"""Streamlit views and schema-driven CRUD forms."""
from datetime import date

import streamlit as st
from sqlalchemy import Date, Text
from sqlalchemy.exc import OperationalError

from loswatch.models import Admission, Bed, Doctor, Patient
from loswatch.services import MODELS, ValidationError, active_admissions, delete_record, records, save_record, search_patients


def label(record):
    if isinstance(record, Patient):
        return f"{record.mrn} · {record.name}"
    return f"#{record.id} · {getattr(record, 'name', getattr(record, 'label', getattr(record, 'disease', getattr(record, 'medication', 'Admission'))))}"


def table(items, exclude=()):
    rows = [{column.name: getattr(item, column.name) for column in item.__table__.columns if column.name not in exclude} for item in items]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No records yet.")


def patients_view(session):
    st.header("Patient search & profiles")
    matches = search_patients(session, st.text_input("Search name or MRN"))
    if not matches:
        st.info("No matching patients. Create one in Manage records or seed the demo.")
        return
    patient_id = st.selectbox("Patient", [p.id for p in matches], format_func=lambda key: label(next(p for p in matches if p.id == key)))
    patient = session.get(Patient, patient_id)
    st.subheader(patient.name)
    st.write(f"MRN: {patient.mrn} | Born: {patient.date_of_birth} | Sex: {patient.sex}")
    st.write(f"Assigned doctor: {patient.doctor.name if patient.doctor else 'Unassigned'}")
    active = next((a for a in patient.admissions if a.discharged_on is None), None)
    if active:
        st.write(f"Current bed: {active.bed.label} · {active.bed.ward}")
        st.metric("Current length of stay (days)", (date.today() - active.admitted_on).days)
    else:
        st.write("No active admission.")
    for title, items in (("Diseases / diagnoses", patient.diagnoses), ("Recorded medications", patient.medications), ("Admission history", patient.admissions)):
        st.subheader(title)
        table(items, exclude=("patient_id",))


def doctors_view(session):
    st.header("Doctors & assignments")
    patients = records(session, Patient)
    for doctor in records(session, Doctor):
        with st.expander(f"{doctor.name} · {doctor.specialty}"):
            table([p for p in patients if p.doctor_id == doctor.id], exclude=("doctor_id",))
    st.subheader("Unassigned patients")
    table([p for p in patients if p.doctor_id is None])
    st.caption("Assign or reassign a doctor by updating the Patient record in Manage records.")


def beds_view(session):
    st.header("Bed occupancy")
    occupied = {a.bed_id: a for a in active_admissions(session)}
    rows = []
    for bed in records(session, Bed):
        admission = occupied.get(bed.id)
        patient = session.get(Patient, admission.patient_id) if admission else None
        rows.append({"Bed": bed.label, "Ward": bed.ward, "Status": "Occupied" if admission else "Available", "Patient": label(patient) if patient else "", "Admission": admission.id if admission else None})
    st.dataframe(rows, hide_index=True, width="stretch") if rows else st.info("No beds yet.")
    st.caption("Transfer by updating an Admission's bed. Discharge by setting its discharge date. Only admissions without a discharge date occupy a bed.")


def manage_view(session):
    st.header("Manage records")
    kind = st.selectbox("Record type", list(MODELS))
    model = MODELS[kind]
    items = records(session, model)
    table(items)
    action = st.radio("Action", ("Create", "Update", "Delete"), horizontal=True)
    record = None
    if action != "Create":
        if not items:
            st.info("Create a record first.")
            return
        record_id = st.selectbox("Record", [r.id for r in items], format_func=lambda key: label(next(r for r in items if r.id == key)))
        record = session.get(model, record_id)
    form_key = f"{kind}-{action}-{record.id if record else 'new'}"
    with st.form(form_key):
        if action == "Delete":
            st.warning("Deletion is permanent. Referenced records cannot be deleted.")
            confirmed = st.checkbox("Confirm deletion")
            values = {}
        else:
            values = {}
            for column in model.__table__.columns:
                name = column.name
                if name == "id":
                    continue
                current = getattr(record, name) if record else None
                title = name.replace("_", " ").capitalize()
                key = f"{form_key}-{name}"
                if column.foreign_keys:
                    fk = next(iter(column.foreign_keys))
                    target = next(m for m in MODELS.values() if m.__tablename__ == fk.column.table.name)
                    choices = records(session, target)
                    ids = [r.id for r in choices]
                    options = [None] + ids
                    values[name] = st.selectbox(title, options, index=options.index(current) if current in options else 0, format_func=lambda value, choices=choices: "Select / unassigned" if value is None else label(next(r for r in choices if r.id == value)), key=key)
                elif name == "sex":
                    options = ["Unknown", "Female", "Male", "Other"]
                    values[name] = st.selectbox(title, options, index=options.index(current) if current else 0, key=key)
                elif isinstance(column.type, Date):
                    values[name] = st.date_input(title + (" (optional; clear to leave open)" if column.nullable else ""), value=current if record or column.nullable else date.today(), min_value=date(1900, 1, 1), max_value=date(2100, 12, 31), key=key)
                elif isinstance(column.type, Text):
                    values[name] = st.text_area(title, value=current or "", key=key)
                else:
                    values[name] = st.text_input(title, value=current or "", max_chars=column.type.length, key=key)
        submitted = st.form_submit_button(action)
    if submitted:
        try:
            if action == "Delete":
                if not confirmed:
                    st.error("Confirm deletion before continuing.")
                    return
                delete_record(session, model, record.id)
            else:
                save_record(session, model, values, record.id if record else None)
            st.session_state["notice"] = f"{kind} {action.lower()} succeeded."
            st.rerun()
        except ValidationError as exc:
            st.error(str(exc))
        except OperationalError:
            session.rollback()
            st.error("Database is busy or unavailable. Please retry.")


def dashboard(factory):
    st.title("LOSWatch-AI")
    st.caption("Fictional operations · Historical UCI records · Experimental LOS estimates")
    if notice := st.session_state.pop("notice", None):
        st.success(notice)
    page = st.sidebar.radio("Navigation", ("Patients", "Doctors", "Beds", "Manage records", "Historical records", "Prediction"))
    if page == "Prediction":
        from loswatch.prediction_ui import prediction_view
        prediction_view()
        return
    if page == "Historical records":
        from loswatch.history_ui import historical_view
        historical_view()
        return
    st.caption("Fictional operational records only")
    with factory() as session:
        beds = records(session, Bed)
        occupied = active_admissions(session)
        columns = st.columns(3)
        columns[0].metric("Patients", len(records(session, Patient)))
        columns[1].metric("Occupied beds", len(occupied))
        columns[2].metric("Available beds", len(beds) - len(occupied))
        {"Patients": patients_view, "Doctors": doctors_view, "Beds": beds_view, "Manage records": manage_view}[page](session)
