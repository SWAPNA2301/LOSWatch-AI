"""Experimental manual admission form, isolated from both patient datasets."""
from datetime import date
import os

import streamlit as st

from loswatch.ml.data import HISTORY
from loswatch.prediction import (
    ADMISSION_SOURCES, ADMISSION_TYPES, AGES, MODEL_DIR, PredictionError,
    artifact_signature, load_bundle, predict_los,
)


@st.cache_resource(show_spinner="Loading experimental LOS model…", max_entries=2)
def cached_bundle(directory, signature):
    return load_bundle(directory)


def prediction_view():
    st.header("Experimental LOS prediction")
    st.warning("Experimental use only — not clinically actionable. Do not use this estimate for treatment, discharge or bed-allocation decisions.")
    st.caption("Enter fictional admission details only. This separate form does not fill missing values from synthetic patient profiles or historical encounters. Historical completed encounters retain their actual LOS in Historical records.")
    directory = os.environ.get("LOSWATCH_MODEL_DIR", str(MODEL_DIR))
    try:
        bundle = cached_bundle(directory, artifact_signature(directory))
    except PredictionError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("Prediction unavailable: the model could not be loaded. Other dashboard pages remain available.")
        return

    st.subheader("Saved Random Forest performance")
    metrics = bundle.metrics["models"]["RandomForest"]["test"]
    columns = st.columns(3)
    columns[0].metric("Test MAE (days)", f"{metrics['MAE']:.3f}")
    columns[1].metric("Test RMSE (days)", f"{metrics['RMSE']:.3f}")
    columns[2].metric("Test R²", f"{metrics['R2']:.3f}")
    st.caption("Phase 2.1 · Historical diabetic encounters lasting 1–14 days · Reused patient-disjoint test set. These are aggregate errors, not an individual confidence interval. Negative R² and substantial long-stay underprediction indicate weak predictive performance.")
    st.info("All seven admission inputs must be known. Leave unknown fields blank; prediction will be blocked. Enter 0 visits only when no visits are known to have occurred in the year before this admission.")

    # Regular widgets ensure changing any input immediately removes a prior
    # result. A form would leave an old prediction visible while edits are pending.
    values = {
        "age": st.selectbox("Age band at admission", [None, *AGES], format_func=lambda value: "Unknown / not supplied" if value is None else value, key="prediction_age"),
        "gender": st.selectbox("Gender recorded at admission", [None, "Female", "Male"], format_func=lambda value: "Unknown / not supported" if value is None else value, key="prediction_gender"),
    }
    for name, options in (("admission_type_id", ADMISSION_TYPES), ("admission_source_id", ADMISSION_SOURCES)):
        values[name] = st.selectbox(
            name.removesuffix("_id").replace("_", " ").capitalize(), [None, *options],
            format_func=lambda value, options=options: "Unknown / not supplied" if value is None else f"{value} · {options[value]}",
            key=f"prediction_{name}",
        )
    for name in HISTORY:
        kind = name.removeprefix("number_")
        values[name] = st.text_input(f"{kind.capitalize()} visits in the preceding year", value="", placeholder="Known whole-number count; blank if unknown", key=f"prediction_{name}")
    admitted_on = st.date_input("Known admission date (optional)", value=None, min_value=date(1900, 1, 1), max_value=date.today(), key="prediction_admitted_on")
    st.caption("Leave the date blank if unknown. Admission date is used only for the optional calendar estimate; it is not a model feature.")
    if st.button("Predict experimental total LOS", key="prediction_submit"):
        try:
            result = predict_los(bundle, values, admitted_on)
        except PredictionError as exc:
            st.error(str(exc))
            return
        st.subheader("Experimental estimate")
        st.metric("Predicted total LOS (days)", f"{result.total_days:.2f}")
        st.caption("Total duration from admission, not remaining days. This result is not saved to any patient record.")
        if result.estimated_discharge is None:
            st.info("No discharge date estimated: a known admission date was not supplied.")
        else:
            st.warning(f"Uncertain estimated discharge date: {result.estimated_discharge.isoformat()} — not clinically actionable.")
            st.caption("Calculated as the supplied admission date plus predicted total LOS rounded up to whole calendar days. This is not a scheduled discharge or an individualized confidence estimate.")
            if result.estimated_discharge < date.today():
                st.caption("This estimate falls in the past; it does not establish that a discharge occurred or is due.")
