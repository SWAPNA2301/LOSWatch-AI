"""Optional, opt-in Snowflake integration for de-identified LOSWatch data.

The module is deliberately independent from the SQLite application database.
The Snowflake connector is imported only when an explicit connection is
requested, so the dashboard remains fully functional offline without the
optional dependency installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from loswatch.ml.data import FEATURES, HISTORY, ROOT

SCHEMA_SQL_PATH = ROOT / "sql" / "snowflake_schema.sql"
ADMIN_SETUP_SQL_PATH = ROOT / "sql" / "snowflake_admin_setup.sql"
DOTENV_PATH = ROOT / ".env"
INITIALIZATION_CONFIRMATION = "INITIALIZE LOSWATCH SNOWFLAKE"
DEMO_SYNC_CONFIRMATION = "SYNC SYNTHETIC LOSWATCH DEMO"
SYNTHETIC_BATCH_ID = "synthetic-demo-phase7b-v1"
SNOWFLAKE_TABLES = {
    "historical_encounters": "LOSWATCH_HISTORICAL_ENCOUNTERS",
    "prediction_logs": "LOSWATCH_PREDICTION_LOGS",
    "drift_reports": "LOSWATCH_DRIFT_REPORTS",
    "model_versions": "LOSWATCH_MODEL_VERSIONS",
    "retraining_audit_events": "LOSWATCH_RETRAINING_AUDIT_EVENTS",
}

_ENV_NAMES = {
    "account": "LOSWATCH_SNOWFLAKE_ACCOUNT",
    "user": "LOSWATCH_SNOWFLAKE_USER",
    "password": "LOSWATCH_SNOWFLAKE_PASSWORD",
    "warehouse": "LOSWATCH_SNOWFLAKE_WAREHOUSE",
    "database": "LOSWATCH_SNOWFLAKE_DATABASE",
    "schema": "LOSWATCH_SNOWFLAKE_SCHEMA",
    "role": "LOSWATCH_SNOWFLAKE_ROLE",
    "authenticator": "LOSWATCH_SNOWFLAKE_AUTHENTICATOR",
}
_ENV_ALIASES = {
    "account": ("LOSWATCH_SNOWFLAKE_ORG_ACCOUNT", "LOSWATCH_SNOWFLAKE_ACCOUNT_IDENTIFIER", "SNOWFLAKE_ACCOUNT"),
    "user": ("SNOWFLAKE_USER",),
    "password": ("SNOWFLAKE_PASSWORD",),
    "warehouse": ("SNOWFLAKE_WAREHOUSE",),
    "database": ("SNOWFLAKE_DATABASE",),
    "schema": ("SNOWFLAKE_SCHEMA",),
    "role": ("SNOWFLAKE_ROLE",),
    "authenticator": ("SNOWFLAKE_AUTHENTICATOR",),
}
_REQUIRED_FIELDS = ("account", "user", "warehouse", "database", "schema")
_SUPPORTED_AUTHENTICATORS = {"", "snowflake", "externalbrowser", "oauth_authorization_code"}
_PASSWORD_AUTHENTICATORS = {"", "snowflake"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_OPAQUE_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_FORBIDDEN_SOURCE_COLUMNS = {
    "encounter_id", "patient_nbr", "patient_id", "mrn", "name", "date_of_birth",
    "diagnosis", "diagnoses", "medication", "medications",
}


class SnowflakeError(RuntimeError):
    """Safe, user-facing Snowflake integration error."""

    def __init__(
        self,
        message: str,
        *,
        error_code: int | None = None,
        sqlstate: str | None = None,
        diagnostic_category: str = "connection",
        diagnostic_hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.sqlstate = sqlstate
        self.diagnostic_category = diagnostic_category
        self.diagnostic_hint = diagnostic_hint


class SnowflakeValidationError(SnowflakeError, ValueError):
    """Rejected configuration or batch data."""


@dataclass(frozen=True)
class SnowflakeConfig:
    """Connection settings loaded from environment variables only."""

    account: str = ""
    user: str = ""
    password: str = field(default="", repr=False)
    warehouse: str = ""
    database: str = ""
    schema: str = ""
    role: str | None = None
    authenticator: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, *, dotenv_path: str | Path | None = DOTENV_PATH) -> "SnowflakeConfig":
        values = dict(os.environ if environ is None else environ)
        if dotenv_path is not None:
            for name, value in read_dotenv(dotenv_path).items():
                values.setdefault(name, value)
        for field_name, aliases in _ENV_ALIASES.items():
            canonical_name = _ENV_NAMES[field_name]
            if not values.get(canonical_name):
                for alias in aliases:
                    if values.get(alias):
                        values[canonical_name] = values[alias]
                        break
        return cls(**{name: values.get(env_name, "").strip() or None for name, env_name in _ENV_NAMES.items()})

    def missing_required(self) -> tuple[str, ...]:
        return tuple(name for name in _REQUIRED_FIELDS if not getattr(self, name))

    def validate(self) -> None:
        missing = self.missing_required()
        if missing:
            names = ", ".join(_ENV_NAMES[name] for name in missing)
            raise SnowflakeValidationError(f"Snowflake configuration is incomplete; set: {names}.")
        authenticator = (self.authenticator or "").lower()
        if authenticator not in _SUPPORTED_AUTHENTICATORS:
            raise SnowflakeValidationError("Snowflake authenticator must be empty, snowflake, externalbrowser, or oauth_authorization_code.")
        if authenticator in _PASSWORD_AUTHENTICATORS and not self.password:
            raise SnowflakeValidationError(f"Snowflake password is required unless authenticator is externalbrowser or oauth_authorization_code; set {_ENV_NAMES['password']} or {_ENV_NAMES['authenticator']}.")
        for name in ("database", "schema", "warehouse", "role"):
            value = getattr(self, name)
            if value and not _IDENTIFIER.fullmatch(value):
                raise SnowflakeValidationError(f"Snowflake {name} must be a simple identifier.")

    def connection_kwargs(self) -> dict[str, str]:
        self.validate()
        result = {
            "account": self.account,
            "user": self.user,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
            "application": "LOSWatch-AI",
        }
        authenticator = (self.authenticator or "").lower()
        if authenticator in _PASSWORD_AUTHENTICATORS:
            result["password"] = self.password
        if self.role:
            result["role"] = self.role
        if self.authenticator:
            result["authenticator"] = "OAUTH_AUTHORIZATION_CODE" if authenticator == "oauth_authorization_code" else self.authenticator
        return result

    def safe_summary(self) -> dict[str, Any]:
        """Return displayable status without password or credential values."""
        return {
            "configured": not self.missing_required(),
            "missing": list(self.missing_required()),
            "account_configured": bool(self.account),
            "account_identifier": self.account or None,
            "target": f"{self.database}.{self.schema}" if self.database and self.schema else None,
            "warehouse_configured": bool(self.warehouse),
            "role_configured": bool(self.role),
            "authenticator": self.authenticator or "default",
        }


def read_dotenv(path: str | Path = DOTENV_PATH) -> dict[str, str]:
    """Read simple KEY=VALUE entries without returning or logging secret values.

    Process environment variables take precedence over the local ``.env`` file.
    The file is intentionally never created, modified or displayed by the app.
    """
    path = Path(path)
    if not path.exists():
        return {}
    if not path.is_file():
        raise SnowflakeValidationError("Snowflake .env path is not a regular file.")
    parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise SnowflakeValidationError(f"Invalid Snowflake .env entry on line {line_number}.")
        name, value = (part.strip() for part in line.split("=", 1))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise SnowflakeValidationError(f"Invalid Snowflake .env variable name on line {line_number}.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[name] = value
    return parsed


def dotenv_status(path: str | Path = DOTENV_PATH) -> dict[str, Any]:
    """Return local .env presence and variable names, never their values."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False, "keys": []}
    values = read_dotenv(path)
    return {"path": str(path), "exists": True, "keys": sorted(values)}


@dataclass(frozen=True)
class BatchSyncResult:
    table: str
    batch_id: str
    rows_submitted: int
    idempotency_key: str


def _quote_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise SnowflakeValidationError("Snowflake object names must be simple identifiers.")
    return f'"{value.upper()}"'


def qualified_table(config: SnowflakeConfig, table_key: str) -> str:
    config.validate()
    try:
        table_name = SNOWFLAKE_TABLES[table_key]
    except KeyError as exc:
        raise SnowflakeValidationError(f"Unknown Snowflake batch table: {table_key}") from exc
    return ".".join((_quote_identifier(config.database), _quote_identifier(config.schema), _quote_identifier(table_name)))


def _connector_module():
    try:
        return importlib.import_module("snowflake.connector")
    except ImportError as exc:
        raise SnowflakeError("Snowflake integration is optional; install requirements-snowflake.txt to connect.") from exc


def _connector_error_details(exc: Exception) -> tuple[int | None, str | None, str, str]:
    """Extract only safe connector metadata and classify the failure."""
    raw_code = getattr(exc, "errno", getattr(exc, "error_code", None))
    if isinstance(raw_code, bool):
        error_code = None
    elif isinstance(raw_code, int):
        error_code = raw_code
    elif raw_code is not None and str(raw_code).strip().isdigit():
        error_code = int(str(raw_code).strip())
    else:
        error_code = None
    raw_sqlstate = getattr(exc, "sqlstate", None)
    sqlstate = str(raw_sqlstate).strip().upper() if raw_sqlstate else None
    message = str(exc).lower()

    if error_code == 390190 or "saml identity provider account parameter" in message:
        return (
            error_code,
            sqlstate,
            "saml_account_mismatch",
            "Verify the org-account identifier matches the Snowflake account configured in the SAML identity provider.",
        )
    if error_code in {390100, 390322} or "authentication" in message or "incorrect username or password" in message:
        return error_code, sqlstate, "authentication", "Verify the selected authenticator and the Snowflake/identity-provider user mapping."
    if error_code == 390186 or ("role" in message and ("not granted" in message or "not authorized" in message)):
        return error_code, sqlstate, "role_access", "Verify that the requested role is granted to the login user and usable by the identity provider."
    if (sqlstate or "").startswith("08") or any(token in message for token in ("timed out", "connection refused", "network is unreachable", "name or service not known")):
        return error_code, sqlstate, "network", "Verify endpoint reachability, proxy/firewall settings, and the account identifier."
    return error_code, sqlstate, "connection", "Review the Snowflake connector and account configuration."


def connect(config: SnowflakeConfig | None = None, *, connector_module: Any | None = None):
    """Open a connector session only when explicitly called by an operator."""
    config = config or SnowflakeConfig.from_env()
    config.validate()
    connector = connector_module or _connector_module()
    try:
        return connector.connect(**config.connection_kwargs())
    except Exception as exc:  # connector exceptions vary by installed version
        error_code, sqlstate, category, hint = _connector_error_details(exc)
        raise SnowflakeError(
            f"Snowflake connection failed ({type(exc).__name__}).",
            error_code=error_code,
            sqlstate=sqlstate,
            diagnostic_category=category,
            diagnostic_hint=hint,
        ) from exc


def test_connection(config: SnowflakeConfig | None = None, *, connector_module: Any | None = None) -> dict[str, Any]:
    """Run a read-only connection check and return a secret-free status."""
    connection = None
    try:
        connection = connect(config, connector_module=connector_module)
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT CURRENT_VERSION(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
            row = cursor.fetchone()
        finally:
            cursor.close()
        return {
            "connected": True,
            "version": str(row[0]) if row else "unknown",
            "role": str(row[1]) if row and len(row) > 1 and row[1] else "unknown",
            "warehouse": str(row[2]) if row and len(row) > 2 and row[2] else "unknown",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
        }
    except SnowflakeError as exc:
        result = {"connected": False, "error": str(exc), "checked_utc": datetime.now(timezone.utc).isoformat()}
        if exc.error_code is not None:
            result["error_code"] = exc.error_code
        if exc.sqlstate:
            result["sqlstate"] = exc.sqlstate
        result["diagnostic_category"] = exc.diagnostic_category
        if exc.diagnostic_hint:
            result["diagnostic_hint"] = exc.diagnostic_hint
        return result
    except Exception as exc:
        error_code, sqlstate, category, hint = _connector_error_details(exc)
        return {
            "connected": False,
            "error": f"Snowflake connection check failed ({type(exc).__name__}).",
            "error_code": error_code,
            "sqlstate": sqlstate,
            "diagnostic_category": category,
            "diagnostic_hint": hint,
            "checked_utc": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        if connection is not None:
            connection.close()


def _render_schema_sql(config: SnowflakeConfig) -> list[str]:
    config.validate()
    if not SCHEMA_SQL_PATH.exists():
        raise SnowflakeError(f"Snowflake schema file not found: {SCHEMA_SQL_PATH}")
    database = _quote_identifier(config.database)
    schema = _quote_identifier(config.schema)
    sql = "\n".join(line for line in SCHEMA_SQL_PATH.read_text(encoding="utf-8").splitlines() if not line.strip().startswith("--"))
    statements = sql.split(";")
    rendered = []
    for statement in statements:
        statement = statement.strip()
        if statement:
            rendered.append(statement.replace("{{DATABASE}}", database).replace("{{SCHEMA}}", schema))
    return rendered


def ensure_schema(connection: Any, config: SnowflakeConfig | None = None, *, include_schema: bool = True) -> int:
    """Create the optional schema objects; no data is uploaded by this call."""
    config = config or SnowflakeConfig.from_env()
    statements = _render_schema_sql(config)
    if not include_schema:
        statements = statements[1:]
    cursor = connection.cursor()
    try:
        for statement in statements:
            cursor.execute(statement)
        if hasattr(connection, "commit"):
            connection.commit()
    finally:
        cursor.close()
    return len(statements)


def setup_preview(config: SnowflakeConfig | None = None) -> dict[str, Any]:
    """Describe DDL and target warehouse usage without connecting or executing SQL."""
    config = config or SnowflakeConfig.from_env()
    statements = _render_schema_sql(config)
    return {
        "read_only": True,
        "database": config.database,
        "schema": config.schema,
        "warehouse": config.warehouse,
        "warehouse_action": "Use existing warehouse only; no CREATE WAREHOUSE or billing resource provisioning.",
        "schema_action": "Database/schema must be prepared by a security administrator; application initialization creates tables only.",
        "statement_count": len(statements),
        "objects": list(SNOWFLAKE_TABLES.values()),
        "statements": statements,
    }


def admin_setup_preview(config: SnowflakeConfig | None = None) -> dict[str, Any]:
    """Return reviewed one-time administrator SQL without connecting or executing it."""
    config = config or SnowflakeConfig.from_env()
    config.validate()
    if not ADMIN_SETUP_SQL_PATH.exists():
        raise SnowflakeError(f"Snowflake administrator setup file not found: {ADMIN_SETUP_SQL_PATH}")
    sql = ADMIN_SETUP_SQL_PATH.read_text(encoding="utf-8")
    return {
        "read_only": True,
        "database": "LOSWATCH_AI",
        "schema": "MONITORING",
        "role": "LOSWATCH_APP",
        "warehouse_action": "Replace the reviewed existing-warehouse placeholder manually; no warehouse is created by LOSWatch.",
        "statements": sql,
    }


def initialize_snowflake(connection: Any, config: SnowflakeConfig, confirmation: str) -> dict[str, Any]:
    """Create schema/tables only after an explicit operator confirmation."""
    if confirmation != INITIALIZATION_CONFIRMATION:
        raise SnowflakeValidationError(f"Initialization requires the exact confirmation text: {INITIALIZATION_CONFIRMATION}.")
    count = ensure_schema(connection, config, include_schema=False)
    return {
        "initialized": True,
        "statement_count": count,
        "database": config.database,
        "schema": config.schema,
        "warehouse": config.warehouse,
        "schema_created": False,
        "warehouse_created": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _canonical_hash(record: Mapping[str, Any], fields: Iterable[str]) -> str:
    payload = {field: record.get(field) for field in fields}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _batch_id(records: list[Mapping[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), default=_json_default, allow_nan=False).encode()
    return "batch-" + hashlib.sha256(encoded).hexdigest()[:24]


_TABLE_FIELDS = {
    "historical_encounters": ("record_key", "source_batch_id", "age_band", "gender", "admission_type_id", "admission_source_id", *HISTORY, "actual_los_days", "record_hash"),
    "prediction_logs": ("prediction_key", "source_batch_id", "model_version", "predicted_los_days", "feature_fingerprint", "created_utc", "record_hash"),
    "drift_reports": ("report_key", "source_batch_id", "model_id", "scenario", "created_utc", "reference_size", "current_size", "drift_detected", "report_json", "record_hash"),
    "model_versions": ("model_version", "source_batch_id", "model_id", "status", "created_utc", "selected_model", "metrics_json", "record_hash"),
    "retraining_audit_events": ("event_key", "source_batch_id", "action", "model_version", "timestamp_utc", "approved_by", "details_json", "record_hash"),
}
_KEY_FIELDS = {
    "historical_encounters": "record_key", "prediction_logs": "prediction_key", "drift_reports": "report_key", "model_versions": "model_version", "retraining_audit_events": "event_key",
}
_VARIANT_FIELDS = {"report_json", "metrics_json", "details_json"}


def _normalize_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_batch(table_key: str, records: Iterable[Mapping[str, Any]], *, batch_id: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """Validate a sanitized batch and attach deterministic hashes for MERGE."""
    if table_key not in _TABLE_FIELDS:
        raise SnowflakeValidationError(f"Unknown Snowflake batch table: {table_key}")
    rows = [dict(record) for record in records]
    if not rows:
        return [], batch_id or "empty-batch"
    fields = _TABLE_FIELDS[table_key]
    allowed = set(fields)
    unknown = set().union(*(set(row) for row in rows)) - allowed
    if unknown:
        raise SnowflakeValidationError(f"Batch contains unsupported or identifying fields: {sorted(unknown)}")
    normalized = []
    for row in rows:
        row = {key: _normalize_value(value) for key, value in row.items()}
        supplied_hash = row.pop("record_hash", None)
        key_name = _KEY_FIELDS[table_key]
        key = str(row.get(key_name) or "")
        if not _OPAQUE_KEY.fullmatch(key):
            raise SnowflakeValidationError(f"{key_name} must be a nonempty opaque key.")
        row[key_name] = key
        for required in ("source_batch_id",):
            if not str(row.get(required) or "").strip():
                raise SnowflakeValidationError(f"{required} is required for idempotent sync.")
        if table_key == "historical_encounters":
            for name in HISTORY:
                value = row.get(name)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or int(value) != value):
                    raise SnowflakeValidationError(f"{name} must be a nonnegative whole number or null.")
            los = row.get("actual_los_days")
            if los is not None and (isinstance(los, bool) or not isinstance(los, (int, float)) or not 1 <= los <= 14 or int(los) != los):
                raise SnowflakeValidationError("actual_los_days must be an integer from 1 to 14 or null.")
        if table_key == "prediction_logs":
            prediction = row.get("predicted_los_days")
            if prediction is None or isinstance(prediction, bool) or not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
                raise SnowflakeValidationError("predicted_los_days must be finite.")
        if table_key in {"drift_reports", "model_versions", "retraining_audit_events"}:
            json_field = {"drift_reports": "report_json", "model_versions": "metrics_json", "retraining_audit_events": "details_json"}[table_key]
            if not isinstance(row.get(json_field), str):
                raise SnowflakeValidationError(f"{json_field} must be serialized JSON.")
            try:
                json.loads(row[json_field])
            except (TypeError, json.JSONDecodeError) as exc:
                raise SnowflakeValidationError(f"{json_field} must contain valid JSON.") from exc
        row["record_hash"] = _canonical_hash(row, fields[:-1])
        if supplied_hash is not None and supplied_hash != row["record_hash"]:
            raise SnowflakeValidationError("record_hash does not match the sanitized batch record.")
        normalized.append({field: row.get(field) for field in fields})
    keys = [row[_KEY_FIELDS[table_key]] for row in normalized]
    if len(keys) != len(set(keys)):
        raise SnowflakeValidationError("A batch contains duplicate idempotency keys.")
    source_batch_ids = {str(row.get("source_batch_id")) for row in normalized if row.get("source_batch_id")}
    resolved_batch_id = batch_id or (next(iter(source_batch_ids)) if len(source_batch_ids) == 1 else _batch_id(normalized))
    return normalized, resolved_batch_id


def _safe_frame_value(value: Any) -> Any:
    value = _normalize_value(value)
    return value


def historical_records(frame: pd.DataFrame, *, batch_id: str | None = None, include_actual_los: bool = True) -> list[dict[str, Any]]:
    """Convert an already de-identified feature frame; identifiers are rejected."""
    forbidden = _FORBIDDEN_SOURCE_COLUMNS & {str(column).lower() for column in frame.columns}
    if forbidden:
        raise SnowflakeValidationError("Historical sync requires a de-identified frame; identifying/raw columns were supplied.")
    required = set(FEATURES) - {"actual_los_days"}
    if not required.issubset(frame.columns):
        raise SnowflakeValidationError(f"De-identified historical frame is missing: {sorted(required - set(frame.columns))}")
    clean = frame.reset_index(drop=True)
    result = []
    for index, source in clean.iterrows():
        values = {name: _safe_frame_value(source[name]) for name in FEATURES}
        values["age_band"] = values.pop("age")
        if not include_actual_los:
            actual = None
        else:
            actual = _safe_frame_value(source.get("actual_los_days"))
        key_payload = {**values, "actual_los_days": actual, "row_number": index}
        record_key = "row-" + hashlib.sha256(json.dumps(key_payload, sort_keys=True, default=_json_default).encode()).hexdigest()[:32]
        result.append({"record_key": record_key, "source_batch_id": batch_id or "", **values, "actual_los_days": actual})
    if not batch_id:
        calculated = _batch_id(result)
        for row in result:
            row["source_batch_id"] = calculated
    return validate_batch("historical_encounters", result, batch_id=batch_id)[0]


def drift_report_records(report: Mapping[str, Any], *, batch_id: str | None = None) -> list[dict[str, Any]]:
    """Prepare one drift-report row without copying raw batch records."""
    if not report.get("model_id") or not report.get("created_utc"):
        raise SnowflakeValidationError("Drift report must include model_id and created_utc.")
    report_json = json.dumps(dict(report), sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = "drift-" + hashlib.sha256(report_json.encode()).hexdigest()[:32]
    source = batch_id or "report-" + hashlib.sha256(report_json.encode()).hexdigest()[:24]
    row = {"report_key": key, "source_batch_id": source, "model_id": str(report["model_id"]), "scenario": str(report.get("scenario", "observed")), "created_utc": str(report["created_utc"]), "reference_size": int(report.get("sample_sizes", {}).get("reference", 0)), "current_size": int(report.get("sample_sizes", {}).get("current", 0)), "drift_detected": bool(report.get("drift_detected", False)), "report_json": report_json}
    return validate_batch("drift_reports", [row], batch_id=source)[0]


def model_version_records(metrics: Mapping[str, Any], *, batch_id: str | None = None, status: str = "active") -> list[dict[str, Any]]:
    version = str(metrics.get("model_version") or metrics.get("selected_trial") or "")
    if not version:
        raise SnowflakeValidationError("Model metrics must include model_version or selected_trial.")
    serialized = json.dumps(dict(metrics), sort_keys=True, separators=(",", ":"), allow_nan=False)
    source = batch_id or "model-" + hashlib.sha256(serialized.encode()).hexdigest()[:24]
    row = {"model_version": version, "source_batch_id": source, "model_id": str(metrics.get("selected_model", "unknown")), "status": status, "created_utc": str(metrics.get("created_utc", datetime.now(timezone.utc).isoformat())), "selected_model": str(metrics.get("selected_trial", metrics.get("selected_model", "unknown"))), "metrics_json": serialized}
    return validate_batch("model_versions", [row], batch_id=source)[0]


def retraining_audit_records(events: Iterable[Mapping[str, Any]], *, batch_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for index, event in enumerate(events):
        serialized = json.dumps(dict(event.get("details", {})), sort_keys=True, separators=(",", ":"), allow_nan=False)
        timestamp = str(event.get("timestamp_utc", ""))
        action = str(event.get("action", ""))
        version = str(event.get("model_version", "unknown"))
        key = "audit-" + hashlib.sha256(f"{timestamp}|{action}|{version}|{index}".encode()).hexdigest()[:32]
        source = batch_id or "audit-" + hashlib.sha256(serialized.encode()).hexdigest()[:24]
        rows.append({"event_key": key, "source_batch_id": source, "action": action, "model_version": version, "timestamp_utc": timestamp, "approved_by": str(event.get("approved_by", "")), "details_json": serialized})
    return validate_batch("retraining_audit_events", rows, batch_id=batch_id)[0]


def preview_batch(table_key: str, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows, batch_id = validate_batch(table_key, records)
    return {"table": SNOWFLAKE_TABLES.get(table_key, table_key), "table_key": table_key, "rows": len(rows), "batch_id": batch_id, "columns": list(_TABLE_FIELDS.get(table_key, ())), "upload_performed": False}


def synthetic_demo_batches() -> dict[str, list[dict[str, Any]]]:
    """Build deterministic records containing no real identifiers or UCI rows."""
    historical = pd.DataFrame([
        {"age": "[30-40)", "gender": "Female", "admission_type_id": 1, "admission_source_id": 7, "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 0, "actual_los_days": 3},
        {"age": "[60-70)", "gender": "Male", "admission_type_id": 2, "admission_source_id": 1, "number_outpatient": 2, "number_emergency": 0, "number_inpatient": 1, "actual_los_days": 7},
    ], columns=[*FEATURES, "actual_los_days"])
    historical_rows = historical_records(historical, batch_id=SYNTHETIC_BATCH_ID)
    drift = {
        "schema_version": "phase7b.demo.v1",
        "model_id": "synthetic-demo-model",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "scenario": "Synthetic demo: no real patient data",
        "sample_sizes": {"reference": 2, "current": 2},
        "drift_detected": False,
        "features": {},
    }
    metrics = {
        "schema_version": "phase7b.demo.v1",
        "model_version": "synthetic-demo-model-v1",
        "selected_model": "SyntheticDemo",
        "selected_trial": "synthetic_demo",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "metrics": {"MAE": 0.0, "RMSE": 0.0},
    }
    events = [{
        "action": "synthetic_demo_sync",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "model_version": "synthetic-demo-model-v1",
        "approved_by": "demo-operator",
        "details": {"purpose": "Phase 7B connectivity test", "real_patient_data": False},
    }]
    predictions = [
        {"prediction_key": "demo-prediction-001", "source_batch_id": SYNTHETIC_BATCH_ID, "model_version": "synthetic-demo-model-v1", "predicted_los_days": 3.5, "feature_fingerprint": "synthetic-feature-fingerprint-001", "created_utc": "2026-01-01T00:00:00+00:00"},
        {"prediction_key": "demo-prediction-002", "source_batch_id": SYNTHETIC_BATCH_ID, "model_version": "synthetic-demo-model-v1", "predicted_los_days": 6.5, "feature_fingerprint": "synthetic-feature-fingerprint-002", "created_utc": "2026-01-01T00:00:00+00:00"},
    ]
    return {
        "historical_encounters": historical_rows,
        "prediction_logs": validate_batch("prediction_logs", predictions, batch_id=SYNTHETIC_BATCH_ID)[0],
        "drift_reports": drift_report_records(drift, batch_id=SYNTHETIC_BATCH_ID),
        "model_versions": model_version_records(metrics, batch_id=SYNTHETIC_BATCH_ID, status="demo"),
        "retraining_audit_events": retraining_audit_records(events, batch_id=SYNTHETIC_BATCH_ID),
    }


def query_batch_counts(connection: Any, config: SnowflakeConfig, batch_id: str) -> dict[str, int]:
    """Read back counts for one source batch using bound parameters."""
    if batch_id != SYNTHETIC_BATCH_ID:
        raise SnowflakeValidationError("Query-back is restricted to the synthetic Phase 7B batch.")
    cursor = connection.cursor()
    counts: dict[str, int] = {}
    try:
        for table_key in SNOWFLAKE_TABLES:
            table = qualified_table(config, table_key)
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE SOURCE_BATCH_ID = %(source_batch_id)s", {"source_batch_id": batch_id})
            row = cursor.fetchone()
            counts[table_key] = int(row[0]) if row else 0
    finally:
        cursor.close()
    return counts


def sync_synthetic_demo(connection: Any, config: SnowflakeConfig, confirmation: str) -> dict[str, Any]:
    """Synchronize only built-in synthetic rows after explicit confirmation."""
    if confirmation != DEMO_SYNC_CONFIRMATION:
        raise SnowflakeValidationError(f"Demo sync requires the exact confirmation text: {DEMO_SYNC_CONFIRMATION}.")
    batches = synthetic_demo_batches()
    results = {}
    for table_key, rows in batches.items():
        if any(row.get("source_batch_id") != SYNTHETIC_BATCH_ID for row in rows):
            raise SnowflakeValidationError("Synthetic demo batch identity check failed.")
        result = sync_batch(connection, config, table_key, rows, batch_id=SYNTHETIC_BATCH_ID)
        results[table_key] = {"submitted": result.rows_submitted, "batch_id": result.batch_id}
    counts = query_batch_counts(connection, config, SYNTHETIC_BATCH_ID)
    for table_key, result in results.items():
        result["stored_count"] = counts[table_key]
        result["duplicate_safe"] = counts[table_key] == result["submitted"]
    return {"synthetic_only": True, "batch_id": SYNTHETIC_BATCH_ID, "tables": results, "query_back_counts": counts}


def _merge_sql(table: str, fields: tuple[str, ...], key_field: str) -> str:
    sql_columns = ", ".join(field.upper() for field in fields)
    source_columns = ", ".join(f"%({field})s AS {field.upper()}" + (f"" if field not in _VARIANT_FIELDS else "") for field in fields)
    source_columns = source_columns.replace("%(report_json)s AS REPORT_JSON", "PARSE_JSON(%(report_json)s) AS REPORT_JSON")
    source_columns = source_columns.replace("%(metrics_json)s AS METRICS_JSON", "PARSE_JSON(%(metrics_json)s) AS METRICS_JSON")
    source_columns = source_columns.replace("%(details_json)s AS DETAILS_JSON", "PARSE_JSON(%(details_json)s) AS DETAILS_JSON")
    updates = ", ".join(f"target.{field.upper()} = source.{field.upper()}" for field in fields if field not in {key_field, "record_hash"})
    updates += ", target.RECORD_HASH = source.RECORD_HASH, target.SYNCED_AT = CURRENT_TIMESTAMP()"
    values = ", ".join(f"source.{field.upper()}" for field in fields)
    return f"MERGE INTO {table} AS target USING (SELECT {source_columns}) AS source ON target.{key_field.upper()} = source.{key_field.upper()} WHEN MATCHED THEN UPDATE SET {updates} WHEN NOT MATCHED THEN INSERT ({sql_columns}) VALUES ({values})"


def sync_batch(connection: Any, config: SnowflakeConfig, table_key: str, records: Iterable[Mapping[str, Any]], *, batch_id: str | None = None) -> BatchSyncResult:
    """Validate and idempotently MERGE a sanitized batch using bound parameters."""
    rows, resolved_batch_id = validate_batch(table_key, records, batch_id=batch_id)
    if not rows:
        return BatchSyncResult(table_key, resolved_batch_id, 0, "empty")
    table = qualified_table(config, table_key)
    fields = _TABLE_FIELDS[table_key]
    query = _merge_sql(table, fields, _KEY_FIELDS[table_key])
    cursor = connection.cursor()
    try:
        cursor.executemany(query, rows)
        if hasattr(connection, "commit"):
            connection.commit()
    except Exception as exc:
        if hasattr(connection, "rollback"):
            connection.rollback()
        raise SnowflakeError(f"Snowflake batch sync failed ({type(exc).__name__}).") from exc
    finally:
        cursor.close()
    return BatchSyncResult(table_key, resolved_batch_id, len(rows), "record-key-merge")
