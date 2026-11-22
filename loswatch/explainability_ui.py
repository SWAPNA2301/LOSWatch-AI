"""Streamlit UI for Phase 6 explainability."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from loswatch.explainability import (
    ExplanationError,
    LIMITATIONS,
    MAX_EXPLANATION_ROWS,
    combine_explanation_reports,
    explain_batch,
    load_active_bundle,
    load_validation_sample,
    save_report,
)
from loswatch.ml.data import HISTORY
from loswatch.prediction import ADMISSION_SOURCES, ADMISSION_TYPES, AGES, load_bundle


@st.cache_resource(show_spinner="Loading approved explanation model…", max_entries=2)
def cached_explanation_bundle(directory, signature):
    return load_bundle(directory)


def explainability_view():
    st.header("Explainable LOS prediction")
    st.warning("Experimental use only. SHAP explanations are associative, not causal, and must not guide treatment, discharge or bed-allocation decisions.")
    try:
        bundle, directory, signature = load_active_bundle()
        bundle = cached_explanation_bundle(str(directory), signature)
    except ExplanationError as exc:
        st.error(str(exc))
        return
    st.caption(f"Approved model version: {bundle.metrics.get('model_version', bundle.metrics.get('selected_trial', 'unknown'))} · admission-time inputs only")
    values = {
        "age": st.selectbox("Age band at admission", [None, *AGES], format_func=lambda value: "Unknown / not supplied" if value is None else value, key="explain_age"),
        "gender": st.selectbox("Gender recorded at admission", [None, "Female", "Male"], format_func=lambda value: "Unknown / not supported" if value is None else value, key="explain_gender"),
    }
    for name, options in (("admission_type_id", ADMISSION_TYPES), ("admission_source_id", ADMISSION_SOURCES)):
        values[name] = st.selectbox(name.removesuffix("_id").replace("_", " ").capitalize(), [None, *options], format_func=lambda value, options=options: "Unknown / not supplied" if value is None else f"{value} · {options[value]}", key=f"explain_{name}")
    for name in HISTORY:
        values[name] = st.number_input(f"{name.removeprefix('number_').capitalize()} visits in preceding year", min_value=0, value=0, step=1, key=f"explain_{name}")
    if st.button("Explain admission", key="explain_submit"):
        try:
            raw = pd.DataFrame([values])
            patient_report = explain_batch(bundle, raw, max_rows=1)
            validation, validation_metadata = load_validation_sample()
            global_report = explain_batch(bundle, validation, max_rows=MAX_EXPLANATION_ROWS)
            report = combine_explanation_reports(patient_report, global_report, validation_metadata)
            report_path = save_report(report)
            st.session_state["explanation_report"] = report
            st.session_state["explanation_report_path"] = str(report_path)
        except ExplanationError as exc:
            st.error(str(exc))
    report = st.session_state.get("explanation_report")
    if not report:
        return
    st.metric("Predicted total LOS", f"{report['predictions_days'][0]:.2f} days")
    st.caption(f"Patient SHAP reconstruction error: {report['max_reconstruction_error']:.3g}; sample cap {MAX_EXPLANATION_ROWS} rows")
    patient = report["patient_contributions"][0]
    contribution_rows = sorted(({"Feature": row["label"], "Contribution (days)": row["contribution"], "Model feature": row["feature"]} for row in patient.get("labeled_contributions", [])), key=lambda row: abs(row["Contribution (days)"]), reverse=True)
    st.subheader("Patient-level feature contributions")
    st.caption("Signed SHAP contributions: positive values increase the prediction relative to the model baseline; negative values decrease it.")
    patient_frame = pd.DataFrame(contribution_rows)
    st.dataframe(patient_frame, hide_index=True, width="stretch")
    if not patient_frame.empty:
        signed = patient_frame.sort_values("Contribution (days)")
        signed_chart = alt.Chart(signed).mark_bar().encode(
            x=alt.X("Contribution (days):Q", title="Contribution to predicted LOS (days)"),
            y=alt.Y("Feature:N", sort=None, title=None),
            color=alt.condition("datum['Contribution (days)'] >= 0", alt.value("#c2410c"), alt.value("#2563eb")),
            tooltip=["Feature", "Model feature", "Contribution (days)"],
        )
        st.altair_chart(signed_chart, use_container_width=True)
    st.subheader("Global feature importance")
    st.caption(f"Mean(|SHAP|) across {report['global_sample_size']} patient-disjoint validation records ({report['global_patient_count']} patients); partition: {report['global_partition']}.")
    importance = pd.DataFrame(report["global_importance"]).head(15).sort_values("mean_abs_shap", ascending=True)
    importance_table = importance[["label", "feature", "mean_abs_shap", "mean_shap"]].rename(columns={"label": "Feature", "feature": "Model feature", "mean_abs_shap": "Mean |SHAP| (days)", "mean_shap": "Mean signed SHAP (days)"})
    st.dataframe(importance_table.sort_values("Mean |SHAP| (days)", ascending=False), hide_index=True, width="stretch")
    importance_chart = alt.Chart(importance).mark_bar().encode(
        x=alt.X("mean_abs_shap:Q", title="Mean |SHAP| (days)"),
        y=alt.Y("label:N", sort=None, title=None),
        tooltip=["label", "feature", "mean_abs_shap", "mean_shap"],
    )
    st.altair_chart(importance_chart, use_container_width=True)
    st.download_button("Download explanation report", json.dumps(report, indent=2), file_name="loswatch_explanation_report.json", mime="application/json")
    st.caption(" · ".join(LIMITATIONS))
    st.caption(f"Versioned report saved to {st.session_state['explanation_report_path']}")
