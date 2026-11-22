"""Streamlit UI for Phase 5 retraining governance."""
import json
from pathlib import Path

import streamlit as st

from loswatch.drift import REPORT_DIR
from loswatch.retraining import DEFAULT_CURRENT, DEFAULT_DATA, DEFAULT_OUTPUT, REGISTRY_DIR, active_model_dir, evaluate_labeled_batch, promote_candidate, rollback_model, retraining_recommendation, train_candidates
from loswatch.ui_theme import page_header, section_header


def _latest_drift():
    files = sorted(REPORT_DIR.glob("*.json")) if REPORT_DIR.exists() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def retraining_view():
    page_header("ML governance", "Retraining & model governance", "Evaluate, compare and promote candidates with explicit operator approval.")
    st.caption("Phase 5 · offline candidates · explicit approval required · no cloud services")
    data_path = st.text_input("Labeled LOS batch CSV", str(DEFAULT_DATA))
    current = active_model_dir()
    current_status = st.columns(2)
    current_status[0].metric("Active model", current.name)
    current_status[1].metric("Candidate policy", "Approval required")
    performance = None
    if st.button("Evaluate labeled batch"):
        try:
            import pandas as pd
            performance = evaluate_labeled_batch(current, pd.read_csv(data_path, low_memory=False))
            st.session_state["retraining_performance"] = performance
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    performance = st.session_state.get("retraining_performance", performance)
    drift = _latest_drift()
    recommendation = retraining_recommendation(drift, performance)
    st.metric("Retraining recommendation", "Recommended" if recommendation["recommend_retraining"] else "Not triggered")
    if recommendation["reasons"]:
        st.warning("; ".join(recommendation["reasons"]))
    if performance:
        st.json(performance)
    output = st.text_input("Candidate output directory", str(DEFAULT_OUTPUT))
    if st.button("Train candidate models"):
        try:
            result = train_candidates(data_path, output, current)
            st.session_state["candidate_result"] = result
            st.success(f"Candidate {result['model_version']} trained.")
        except (OSError, ValueError, FileExistsError) as exc:
            st.error(str(exc))
    candidate = st.session_state.get("candidate_result")
    if candidate:
        section_header("Candidate vs current", "Both models are evaluated on the identical patient-disjoint validation set.")
        st.json({"candidate": candidate["candidate"], "current": candidate["current"]})
        approver = st.text_input("Approver identity", key="promotion_approver")
        reason = st.text_area("Approval reason", key="promotion_reason")
        if st.button("Approve and promote candidate"):
            try:
                destination = promote_candidate(candidate["output"], REGISTRY_DIR, approver, reason)
                st.success(f"Promoted {destination.name}. Prediction page now uses the approved version.")
            except (OSError, ValueError, FileExistsError) as exc:
                st.error(str(exc))
    pointer = REGISTRY_DIR / "current.json"
    if pointer.exists():
        current_version = json.loads(pointer.read_text(encoding="utf-8"))["model_version"]
        rollback_version = st.text_input("Registered version to activate", current_version)
        if st.button("Approve rollback"):
            try:
                rollback_model(rollback_version, REGISTRY_DIR, st.session_state.get("promotion_approver", ""), "Operator-requested rollback")
                st.success(f"Activated {rollback_version}.")
            except (OSError, ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
