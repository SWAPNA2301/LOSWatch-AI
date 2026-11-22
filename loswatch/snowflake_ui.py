"""Streamlit status and preview-only controls for optional Snowflake sync."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from loswatch.drift import REPORT_DIR
from loswatch.retraining import REGISTRY_DIR, active_model_dir
from loswatch.snowflake import (
    DEMO_SYNC_CONFIRMATION,
    INITIALIZATION_CONFIRMATION,
    SnowflakeConfig,
    SnowflakeError,
    admin_setup_preview,
    connect,
    drift_report_records,
    dotenv_status,
    initialize_snowflake,
    model_version_records,
    preview_batch,
    retraining_audit_records,
    setup_preview,
    sync_synthetic_demo,
    test_connection,
)
from loswatch.ui_theme import page_header, section_header


def _latest_json(directory: Path, pattern: str) -> dict | None:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def _safe_previews() -> dict[str, tuple[str, list[dict]]]:
    previews: dict[str, tuple[str, list[dict]]] = {}
    drift = _latest_json(REPORT_DIR, "*.json")
    if drift:
        previews["Latest drift report"] = ("drift_reports", drift_report_records(drift))
    metrics_path = active_model_dir() / "metrics.json"
    if metrics_path.exists():
        previews["Active model metrics"] = ("model_versions", model_version_records(json.loads(metrics_path.read_text(encoding="utf-8"))))
    audit_path = REGISTRY_DIR / "audit.jsonl"
    if audit_path.exists():
        events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if events:
            previews["Model registry audit"] = ("retraining_audit_events", retraining_audit_records(events))
    return previews


def snowflake_view():
    page_header("Cloud integration", "Snowflake integration", "Optional, manual and de-identified synchronization with local SQLite remaining the default.")
    st.caption("Phase 7A · optional · de-identified batches only · manual actions only")
    st.info("Snowflake is disabled by default. This page never uploads automatically and never reads credentials from the dashboard form. Historical sync requires a separately prepared de-identified frame; raw UCI and SQLite records are not eligible by default.")
    config = SnowflakeConfig.from_env()
    status = config.safe_summary()
    st.caption("Credentials are read from environment variables or the local ignored .env file; values are never displayed.")
    st.json({"dotenv": dotenv_status(), "configuration": status})
    if status["configured"]:
        st.success(f"Configuration is present for {status['target']} (connection not opened automatically).")
    else:
        st.warning("Snowflake is not configured. Set the documented LOSWATCH_SNOWFLAKE_* environment variables to enable an explicit connection check.")
    if st.button("Check Snowflake connection", disabled=not status["configured"]):
        st.session_state["snowflake_connection_status"] = test_connection(config)
    connection_status = st.session_state.get("snowflake_connection_status")
    if connection_status:
        if connection_status.get("connected"):
            st.success(
                "Connection succeeded; "
                f"role={connection_status.get('role', 'unknown')}, "
                f"warehouse={connection_status.get('warehouse', 'unknown')}."
            )
        else:
            st.error(connection_status.get("error", "Snowflake connection failed."))
        st.json(connection_status)

    section_header("Read-only setup preview", "Reviewed SQL is displayed for manual execution; this page does not execute DDL.")
    if status["configured"]:
        try:
            admin = admin_setup_preview(config)
            with st.expander("Show reviewed one-time administrator SQL (manual only)"):
                st.json({key: value for key, value in admin.items() if key != "statements"})
                st.code(admin["statements"], language="sql")
            setup = setup_preview(config)
            with st.expander("Show planned application table statements"):
                st.json({key: value for key, value in setup.items() if key != "statements"})
                st.code(";\n\n".join(setup["statements"]), language="sql")
        except SnowflakeError as exc:
            st.error(str(exc))
    else:
        st.info("Complete configuration is required before a schema/warehouse setup preview can be rendered.")

    section_header("Manual initialization", "Requires the exact confirmation text and an already configured restricted role.")
    init_text = st.text_input("Type the exact initialization confirmation", placeholder=INITIALIZATION_CONFIRMATION, key="snowflake_init_confirmation")
    if st.button("Initialize schema and tables", disabled=not status["configured"]):
        connection = None
        try:
            connection = connect(config)
            result = initialize_snowflake(connection, config, init_text)
            st.session_state["snowflake_initialized"] = result
            st.success("Application tables initialized in the existing schema. No database, schema or warehouse was created, and no data was uploaded.")
        except SnowflakeError as exc:
            st.error(str(exc))
        finally:
            if connection is not None:
                connection.close()
    if st.session_state.get("snowflake_initialized"):
        st.json(st.session_state["snowflake_initialized"])

    section_header("Synthetic demo synchronization", "Only deterministic built-in demo rows are eligible.")
    st.caption("Only deterministic built-in demo rows are eligible. Real patients, raw UCI records and SQLite records are not used.")
    sync_text = st.text_input("Type the exact synthetic-sync confirmation", placeholder=DEMO_SYNC_CONFIRMATION, key="snowflake_demo_confirmation")
    if st.button("Sync synthetic demo", disabled=not status["configured"]):
        connection = None
        try:
            connection = connect(config)
            result = sync_synthetic_demo(connection, config, sync_text)
            st.session_state["snowflake_sync_status"] = result
            st.success("Synthetic demo synchronization completed and query-back counts were verified.")
        except SnowflakeError as exc:
            st.error(str(exc))
        finally:
            if connection is not None:
                connection.close()
    if st.session_state.get("snowflake_sync_status"):
        st.json(st.session_state["snowflake_sync_status"])

    section_header("Safe batch preview", "Preview payloads locally before any explicit sync action.")
    previews = _safe_previews()
    if not previews:
        st.info("No sanitized drift, model-version or registry-audit batch is available for preview.")
        return
    selected = st.selectbox("Local batch", list(previews))
    table_key, records = previews[selected]
    if st.button("Preview sync payload"):
        try:
            st.json(preview_batch(table_key, records))
            st.caption("Preview only: no Snowflake write was performed.")
        except SnowflakeError as exc:
            st.error(str(exc))
