"""Streamlit UI for Phase 6 explainability."""
import json

import pandas as pd
import plotly.graph_objects as go
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
from loswatch.ui_theme import page_header, section_header


@st.cache_resource(show_spinner="Loading approved explanation model…", max_entries=2)
def cached_explanation_bundle(directory, signature):
    return load_bundle(directory)


def explainability_view():
    page_header("ML analytics", "Explainable LOS prediction", "Inspect associative SHAP contributions for the approved admission-time model.")
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
    section_header("Patient-level feature contributions", "Signed SHAP values show how each input moves the prediction relative to the model baseline.")
    st.caption("Signed SHAP contributions: positive values increase the prediction relative to the model baseline; negative values decrease it.")
    patient_frame = pd.DataFrame(contribution_rows)
    st.dataframe(patient_frame, hide_index=True, width="stretch")
    if not patient_frame.empty:
        signed = patient_frame.sort_values("Contribution (days)")
        signed_chart = go.Figure(go.Bar(x=signed["Contribution (days)"], y=signed["Feature"], orientation="h", marker_color=["#c2410c" if value >= 0 else "#2563eb" for value in signed["Contribution (days)"]], customdata=signed[["Model feature"]], hovertemplate="%{y}<br>Contribution: %{x:.3f} days<br>Model feature: %{customdata[0]}<extra></extra>"))
        signed_chart.update_layout(height=max(280, 26 * len(signed)), xaxis_title="Contribution to predicted LOS (days)", yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=12, r=12, t=18, b=24), font=dict(family="Inter, Segoe UI, sans-serif", color="#122033"))
        st.plotly_chart(signed_chart, width="stretch", config={"displayModeBar": False})
    section_header("Global feature importance", "Mean absolute SHAP across representative patient-disjoint validation records.")
    st.caption(f"Mean(|SHAP|) across {report['global_sample_size']} patient-disjoint validation records ({report['global_patient_count']} patients); partition: {report['global_partition']}.")
    importance = pd.DataFrame(report["global_importance"]).head(15).sort_values("mean_abs_shap", ascending=True)
    importance_table = importance[["label", "feature", "mean_abs_shap", "mean_shap"]].rename(columns={"label": "Feature", "feature": "Model feature", "mean_abs_shap": "Mean |SHAP| (days)", "mean_shap": "Mean signed SHAP (days)"})
    st.dataframe(importance_table.sort_values("Mean |SHAP| (days)", ascending=False), hide_index=True, width="stretch")
    importance_chart = go.Figure(go.Bar(x=importance["mean_abs_shap"], y=importance["label"], orientation="h", marker_color="#2563eb", customdata=importance[["feature", "mean_shap"]], hovertemplate="%{y}<br>Mean |SHAP|: %{x:.3f} days<br>Model feature: %{customdata[0]}<br>Mean signed SHAP: %{customdata[1]:.3f}<extra></extra>"))
    importance_chart.update_layout(height=max(360, 25 * len(importance)), xaxis_title="Mean |SHAP| (days)", yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=12, r=12, t=18, b=24), font=dict(family="Inter, Segoe UI, sans-serif", color="#122033"))
    st.plotly_chart(importance_chart, width="stretch", config={"displayModeBar": False})
    st.download_button("Download explanation report", json.dumps(report, indent=2), file_name="loswatch_explanation_report.json", mime="application/json")
    st.caption(" · ".join(LIMITATIONS))
    st.caption(f"Versioned report saved to {st.session_state['explanation_report_path']}")
