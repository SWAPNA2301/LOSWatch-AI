"""Streamlit views and schema-driven CRUD forms."""
from collections import Counter
from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import Date, Text
from sqlalchemy.exc import OperationalError

from loswatch.models import Admission, Bed, Doctor, Patient
from loswatch.services import MODELS, ValidationError, active_admissions, delete_record, records, save_record, search_patients
from loswatch.ui_theme import apply_theme, page_header, section_header, sidebar_brand


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


def _plotly_config(fig, height=300):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=12, r=12, t=18, b=12), font=dict(family="Inter, Segoe UI, sans-serif", color="#122033"), legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#edf2f7", zeroline=False)
    return fig


def _active_stay_days(admission):
    return max((date.today() - admission.admitted_on).days, 0)


def home_view(session):
    """Hospital overview built solely from the existing local records."""
    page_header("Operations command center", "Hospital overview", "A verified snapshot of current patients, beds and recorded admissions.")
    patients = records(session, Patient)
    doctors = records(session, Doctor)
    beds = records(session, Bed)
    admissions = records(session, Admission)
    active = active_admissions(session)
    occupied_bed_ids = {admission.bed_id for admission in active}
    occupancy = (len(occupied_bed_ids) / len(beds) * 100) if beds else 0

    metrics = st.columns(4)
    metrics[0].metric("Patients", len(patients))
    metrics[1].metric("Active admissions", len(active))
    metrics[2].metric("Bed occupancy", f"{occupancy:.0f}%", f"{len(occupied_bed_ids)} of {len(beds)} beds" if beds else "No beds recorded")
    metrics[3].metric("Doctors", len(doctors))

    section_header("Operational pulse", "Trends and occupancy are calculated from recorded database rows; no forecast is shown.")
    left, right = st.columns((1.35, 1))
    with left:
        counts = Counter(admission.admitted_on for admission in admissions)
        if counts:
            start = min(counts)
            end = max(max(counts), date.today())
            days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
            figure = go.Figure(go.Scatter(x=days, y=[counts.get(day, 0) for day in days], mode="lines+markers", line=dict(color="#0f766e", width=3), marker=dict(size=6)))
            figure.update_layout(title="Recorded admissions", xaxis_title=None, yaxis_title="Admissions")
            st.plotly_chart(_plotly_config(figure, 310), width="stretch", config={"displayModeBar": False})
        else:
            st.info("No admission records are available for a trend yet.")
    with right:
        wards = sorted({bed.ward for bed in beds})
        ward_total = Counter(bed.ward for bed in beds)
        ward_occupied = Counter(bed.ward for bed in beds if bed.id in occupied_bed_ids)
        if wards:
            figure = go.Figure()
            figure.add_bar(name="Occupied", x=wards, y=[ward_occupied[ward] for ward in wards], marker_color="#0f766e")
            figure.add_bar(name="Available", x=wards, y=[ward_total[ward] - ward_occupied[ward] for ward in wards], marker_color="#cbd5e1")
            figure.update_layout(title="Beds by ward", barmode="stack", xaxis_title=None, yaxis_title="Beds")
            st.plotly_chart(_plotly_config(figure, 310), width="stretch", config={"displayModeBar": False})
        else:
            st.info("No beds are available for occupancy analysis yet.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        section_header("Patient mix", "Recorded sex values only.")
        sex_counts = Counter((patient.sex or "Unknown") for patient in patients)
        if sex_counts:
            figure = go.Figure(go.Pie(labels=list(sex_counts), values=list(sex_counts.values()), hole=0.58, marker=dict(colors=["#2563eb", "#0f766e", "#f59e0b", "#94a3b8"])))
            figure.update_layout(showlegend=True)
            st.plotly_chart(_plotly_config(figure, 260), width="stretch", config={"displayModeBar": False})
        else:
            st.info("No patient demographics are recorded.")
    with lower_right:
        section_header("Recorded diagnoses", "Most frequent diagnosis labels in the local patient records.")
        diagnoses = Counter(diagnosis.disease for patient in patients for diagnosis in patient.diagnoses)
        if diagnoses:
            top = diagnoses.most_common(8)[::-1]
            figure = go.Figure(go.Bar(x=[count for _, count in top], y=[name for name, _ in top], orientation="h", marker_color="#2563eb"))
            figure.update_layout(xaxis_title="Records", yaxis_title=None)
            st.plotly_chart(_plotly_config(figure, 260), width="stretch", config={"displayModeBar": False})
        else:
            st.info("No diagnoses are recorded.")

    section_header("Current bed board", "Active admissions only.")
    bed_rows = []
    for admission in active:
        bed = session.get(Bed, admission.bed_id)
        patient = session.get(Patient, admission.patient_id)
        bed_rows.append({"Patient": patient.name if patient else "Unknown", "Ward": bed.ward if bed else "Unknown", "Bed": bed.label if bed else "Unknown", "Admitted": admission.admitted_on, "Days since admission": _active_stay_days(admission)})
    st.dataframe(bed_rows, hide_index=True, width="stretch") if bed_rows else st.info("No active admissions are recorded.")


def patients_view(session):
    page_header("Hospital operations", "Patient search & profiles", "Find a patient, review their current stay and inspect recorded history.")
    patients = records(session, Patient)
    search = st.text_input("Search name or MRN", placeholder="e.g. DEMO-0001 or Aarav")
    sex_options = ["All"] + sorted({patient.sex or "Unknown" for patient in patients})
    doctor_options = ["All"] + sorted({patient.doctor.name for patient in patients if patient.doctor})
    filters = st.columns(2)
    selected_sex = filters[0].selectbox("Sex", sex_options)
    selected_doctor = filters[1].selectbox("Assigned doctor", doctor_options)
    matches = search_patients(session, search)
    if selected_sex != "All":
        matches = [patient for patient in matches if (patient.sex or "Unknown") == selected_sex]
    if selected_doctor != "All":
        matches = [patient for patient in matches if patient.doctor and patient.doctor.name == selected_doctor]
    st.caption(f"Showing {len(matches)} of {len(patients)} recorded patients")
    if not matches:
        st.info("No matching patients. Create one in Manage records or seed the demo.")
        return
    patient_id = st.selectbox("Patient", [patient.id for patient in matches], format_func=lambda key: label(next(patient for patient in matches if patient.id == key)))
    patient = session.get(Patient, patient_id)
    st.subheader(patient.name)
    st.write(f"MRN: {patient.mrn} | Born: {patient.date_of_birth} | Sex: {patient.sex}")
    st.write(f"Assigned doctor: {patient.doctor.name if patient.doctor else 'Unassigned'}")
    active = next((admission for admission in patient.admissions if admission.discharged_on is None), None)
    if active:
        st.write(f"Current bed: {active.bed.label} · {active.bed.ward}")
        st.metric("Current length of stay (days)", _active_stay_days(active))
    else:
        st.write("No active admission.")
    for title, items in (("Diseases / diagnoses", patient.diagnoses), ("Recorded medications", patient.medications), ("Admission history", patient.admissions)):
        st.subheader(title)
        table(items, exclude=("patient_id",))


def doctors_view(session):
    page_header("Hospital operations", "Doctors & assignments", "Review specialties and the patients currently assigned to each doctor.")
    patients = records(session, Patient)
    doctors = records(session, Doctor)
    query = st.text_input("Search doctors", placeholder="Name or specialty")
    specialties = ["All"] + sorted({doctor.specialty for doctor in doctors})
    selected_specialty = st.selectbox("Specialty", specialties)
    filtered = [doctor for doctor in doctors if (not query or query.lower() in doctor.name.lower() or query.lower() in doctor.specialty.lower()) and (selected_specialty == "All" or doctor.specialty == selected_specialty)]
    st.caption(f"Showing {len(filtered)} of {len(doctors)} recorded doctors")
    for doctor in filtered:
        with st.expander(f"{doctor.name} · {doctor.specialty}"):
            table([patient for patient in patients if patient.doctor_id == doctor.id], exclude=("doctor_id",))
    st.subheader("Unassigned patients")
    table([patient for patient in patients if patient.doctor_id is None])
    st.caption("Assign or reassign a doctor by updating the Patient record in Manage records.")


def beds_view(session):
    page_header("Hospital operations", "Bed occupancy", "See availability and active admissions by ward using the local bed board.")
    occupied = {admission.bed_id: admission for admission in active_admissions(session)}
    beds = records(session, Bed)
    wards = ["All"] + sorted({bed.ward for bed in beds})
    filters = st.columns(2)
    selected_ward = filters[0].selectbox("Ward", wards)
    selected_status = filters[1].selectbox("Status", ["All", "Occupied", "Available"])
    rows = []
    for bed in beds:
        admission = occupied.get(bed.id)
        status = "Occupied" if admission else "Available"
        if selected_ward != "All" and bed.ward != selected_ward:
            continue
        if selected_status != "All" and status != selected_status:
            continue
        patient = session.get(Patient, admission.patient_id) if admission else None
        rows.append({"Bed": bed.label, "Ward": bed.ward, "Status": status, "Patient": patient.name if patient else "", "Admitted": admission.admitted_on if admission else None})
    st.caption(f"Showing {len(rows)} of {len(beds)} recorded beds")
    st.dataframe(rows, hide_index=True, width="stretch") if rows else st.info("No beds match the selected filters.")
    st.caption("Transfer by updating an Admission's bed. Discharge by setting its discharge date. Only admissions without a discharge date occupy a bed.")


def manage_view(session):
    page_header("Hospital operations", "Manage records", "Create, update or delete local demo records with the existing validation rules.")
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
    apply_theme()
    sidebar_brand()
    st.sidebar.caption("Fictional operational records · Experimental LOS estimates")
    if notice := st.session_state.pop("notice", None):
        st.success(notice)
    st.sidebar.markdown("**Hospital operations**")
    page = st.sidebar.radio("Navigation", ("Home", "Patients", "Doctors", "Beds", "Manage records", "Historical records", "Prediction", "Explainability", "Drift Monitoring", "Retraining", "Snowflake"))
    st.sidebar.markdown("**ML analytics & monitoring**")
    if page == "Prediction":
        from loswatch.prediction_ui import prediction_view
        prediction_view()
        return
    if page == "Historical records":
        from loswatch.history_ui import historical_view
        historical_view()
        return
    if page == "Drift Monitoring":
        from loswatch.drift_ui import drift_view
        drift_view()
        return
    if page == "Retraining":
        from loswatch.retraining_ui import retraining_view
        retraining_view()
        return
    if page == "Explainability":
        from loswatch.explainability_ui import explainability_view
        explainability_view()
        return
    if page == "Snowflake":
        from loswatch.snowflake_ui import snowflake_view
        snowflake_view()
        return
    with factory() as session:
        if page == "Home":
            home_view(session)
            return
        beds = records(session, Bed)
        occupied = active_admissions(session)
        summary = st.columns(3)
        summary[0].metric("Patients", len(records(session, Patient)))
        summary[1].metric("Occupied beds", len(occupied))
        summary[2].metric("Available beds", len(beds) - len(occupied))
        {"Patients": patients_view, "Doctors": doctors_view, "Beds": beds_view, "Manage records": manage_view}[page](session)
