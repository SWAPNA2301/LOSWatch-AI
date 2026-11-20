"""Searchable retrospective UCI view; no operational writes or predictions."""
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from loswatch.history import DIAGNOSES, MEDICATIONS, RAW_DIR, SECTIONS, filter_encounters, mapped_label, parse_mappings, read_encounters


@st.cache_data(show_spinner="Loading historical encounters…", max_entries=2)
def load_history(csv_path, mapping_path, csv_version, mapping_version):
    # File fingerprints in the cache key invalidate the cache after replacement.
    return read_encounters(csv_path), parse_mappings(mapping_path)


def display(value):
    return "Missing" if pd.isna(value) else str(value)


def historical_view():
    st.header("Historical UCI encounters")
    st.caption("Read-only, de-identified 1999–2008 records. Separate from fictional active patients, assignments and bed occupancy. Actual LOS is an observed outcome, not a prediction.")
    directory = Path(os.environ.get("LOSWATCH_HISTORY_DIR", str(RAW_DIR)))
    csv_path, mapping_path = directory / "diabetic_data.csv", directory / "IDS_mapping.csv"
    try:
        csv_stat, mapping_stat = csv_path.stat(), mapping_path.stat()
        frame, mappings = load_history(str(csv_path), str(mapping_path),
                                       (csv_stat.st_mtime_ns, csv_stat.st_size),
                                       (mapping_stat.st_mtime_ns, mapping_stat.st_size))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.info(f"Historical data unavailable: {exc}")
        st.caption("Stage the two supplied CSV files with the historical data command in the README.")
        return
    columns = st.columns(2)
    columns[0].metric("Historical encounters", len(frame))
    columns[1].metric("Historical patients", frame.patient_nbr.nunique())
    encounter = st.text_input("Encounter ID contains")
    patient = st.text_input("Historical patient ID contains")
    diagnosis = st.text_input("Diagnosis code contains (ICD-9-CM)")
    medication = st.selectbox("Medication indicator", [None, *MEDICATIONS], format_func=lambda item: "All medications" if item is None else item)
    status = st.selectbox("Medication status", ["Any", "Recorded use", "No", "Steady", "Up", "Down"], disabled=medication is None)
    matches = filter_encounters(frame, encounter, patient, diagnosis, medication, status)
    st.write(f"{len(matches):,} matching encounters")
    if matches.empty:
        st.info("No historical encounters match these filters.")
        return
    pages = math.ceil(len(matches) / 50)
    filter_key = repr((encounter, patient, diagnosis, medication, status))
    if st.session_state.get("history_filters") != filter_key:
        st.session_state["history_page"] = 1
        st.session_state["history_filters"] = filter_key
    st.session_state["history_page"] = min(st.session_state.get("history_page", 1), pages)
    page = st.number_input("Results page (50 per page)", min_value=1, max_value=pages, step=1, key="history_page")
    page_frame = matches.iloc[(page - 1) * 50:page * 50]
    summary = page_frame[["encounter_id", "patient_nbr", "age", "gender", *DIAGNOSES, "time_in_hospital"]].copy()
    for name in SECTIONS:
        summary[name.removesuffix("_id")] = page_frame[name].map(lambda code: mapped_label(mappings[name], code))
    st.dataframe(summary.rename(columns={"time_in_hospital": "Actual LOS (days)"}), hide_index=True, width="stretch")
    encounter_id = st.selectbox("Historical encounter on this page", page_frame.encounter_id.tolist())
    record = page_frame.loc[page_frame.encounter_id == encounter_id].iloc[0]
    st.subheader(f"Historical encounter {encounter_id}")
    st.write(f"Historical patient: {record.patient_nbr} · Age: {display(record.age)} · Gender: {display(record.gender)}")
    st.metric("Actual LOS (days)", int(record.time_in_hospital))
    st.subheader("Admission and outcome details")
    for name in SECTIONS:
        st.write(f"{name.replace('_id', '').replace('_', ' ').capitalize()}: {display(record[name])} — {mapped_label(mappings[name], record[name])}")
    st.write(f"Readmission outcome: {display(record.get('readmitted'))}")
    st.caption("Discharge disposition and readmission are historical outcomes and are excluded from ML inputs. Calendar admission dates, named doctors and bed assignments are not supplied in this dataset.")
    st.subheader("Recorded diagnoses")
    st.dataframe([{"Position": name, "ICD-9-CM code": display(record[name])} for name in DIAGNOSES], hide_index=True)
    st.caption("Encounter diagnosis codes; their presence at admission is not established. No disease names are inferred from codes.")
    st.subheader("Medication indicators")
    st.dataframe([{"Medication": name, "Indicator": display(record[name])} for name in MEDICATIONS], hide_index=True, width="stretch")
    st.write(f"Medication change: {display(record['change'])} · Diabetes medication: {display(record.diabetesMed)}")
    st.caption("No = not prescribed; Steady = no dosage change; Up/Down = dosage increased/decreased during the encounter. These are recorded indicators, not current orders or admission-time predictors. Missing is distinct from No.")
