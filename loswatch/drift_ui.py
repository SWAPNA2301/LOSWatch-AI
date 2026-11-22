"""Streamlit UI for Phase 4 drift monitoring."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from loswatch.drift import build_batches, analyze, load_reference_data, save_report
from loswatch.ui_theme import page_header, section_header


def drift_view():
    page_header("Monitoring", "Data drift monitoring", "Compare reproducible admission-time batches against the saved reference snapshot.")
    st.caption("Phase 4 · admission-time inputs only · no retraining")
    st.info("Reference data is the saved UCI admission-feature snapshot. Simulated scenarios are reproducible and clearly labeled; they do not alter patients, history, or model artifacts.")
    scenario = st.selectbox("Current batch scenario", ["no_drift", "numeric_shift", "categorical_shift", "missingness_shift", "unseen_category", "combined"], format_func=lambda value: value.replace("_", " ").title())
    size = st.number_input("Batch size", min_value=2, max_value=5000, value=500, step=50)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    if st.button("Run drift comparison"):
        try:
            reference = load_reference_data()
            ref_batch, current_batch, label = build_batches(reference, int(size), int(seed), scenario)
            report = analyze(ref_batch, current_batch, scenario=label)
            path = save_report(report)
            st.session_state["drift_report"] = report
            st.session_state["drift_report_path"] = str(path)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    report = st.session_state.get("drift_report")
    if not report:
        return
    section_header(report["scenario"], "Scenario label and scores are preserved in the versioned report.")
    summary = st.columns(3)
    summary[0].metric("Drift alert", "Detected" if report["drift_detected"] else "No drift")
    summary[1].metric("Reference rows", report["sample_sizes"]["reference"])
    summary[2].metric("Current rows", report["sample_sizes"]["current"])
    st.write(f"Reference: {report['sample_sizes']['reference']} rows · Current: {report['sample_sizes']['current']} rows · α={report['alpha']}")
    if report["alerts"]:
        st.warning("Alerts: " + "; ".join(f"{alert['feature']} ({', '.join(alert['reasons'])})" for alert in report["alerts"]))
    rows = []
    for name, result in report["features"].items():
        rows.append({"Feature": name, "Type": result["type"], "PSI": result["psi"], "KL": result["kl_divergence"], "Wasserstein": result["wasserstein"], "Adjusted p-value": result["adjusted_pvalue"], "Missing Δ": result["missing_rate"]["delta"], "Unseen": ", ".join(result["unseen_categories"]), "Alert": result["drift"]})
    st.dataframe(rows, hide_index=True, width="stretch")
    section_header("Feature distribution scores", "PSI and KL are shown together for the same feature batch.")
    scores = pd.DataFrame(rows).set_index("Feature")[["PSI", "KL"]]
    figure = go.Figure()
    figure.add_bar(name="PSI", x=scores.index, y=scores["PSI"], marker_color="#0f766e")
    figure.add_bar(name="KL divergence", x=scores.index, y=scores["KL"], marker_color="#2563eb")
    figure.update_layout(barmode="group", height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=12, r=12, t=18, b=80), font=dict(family="Inter, Segoe UI, sans-serif", color="#122033"))
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.caption(f"Versioned report saved to {st.session_state['drift_report_path']}")
