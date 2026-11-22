"""Streamlit UI for Phase 4 drift monitoring."""
import streamlit as st
import pandas as pd

from loswatch.drift import build_batches, analyze, load_reference_data, save_report


def drift_view():
    st.header("Data drift monitoring")
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
    st.subheader(report["scenario"])
    st.metric("Drift alert", "Detected" if report["drift_detected"] else "No drift")
    st.write(f"Reference: {report['sample_sizes']['reference']} rows · Current: {report['sample_sizes']['current']} rows · α={report['alpha']}")
    if report["alerts"]:
        st.warning("Alerts: " + "; ".join(f"{alert['feature']} ({', '.join(alert['reasons'])})" for alert in report["alerts"]))
    rows = []
    for name, result in report["features"].items():
        rows.append({"Feature": name, "Type": result["type"], "PSI": result["psi"], "KL": result["kl_divergence"], "Wasserstein": result["wasserstein"], "Adjusted p-value": result["adjusted_pvalue"], "Missing Δ": result["missing_rate"]["delta"], "Unseen": ", ".join(result["unseen_categories"]), "Alert": result["drift"]})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.subheader("Feature distribution scores")
    st.bar_chart(pd.DataFrame(rows).set_index("Feature")[["PSI", "KL"]])
    st.caption(f"Versioned report saved to {st.session_state['drift_report_path']}")
