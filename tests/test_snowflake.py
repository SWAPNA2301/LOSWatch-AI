import json
import re
from pathlib import Path

import pandas as pd
import pytest

from loswatch.ml.data import FEATURES
from loswatch.snowflake import (
    SnowflakeConfig,
    SnowflakeError,
    SnowflakeValidationError,
    admin_setup_preview,
    connect,
    DEMO_SYNC_CONFIRMATION,
    INITIALIZATION_CONFIRMATION,
    ensure_schema,
    historical_records,
    initialize_snowflake,
    preview_batch,
    setup_preview,
    sync_batch,
    sync_synthetic_demo,
    synthetic_demo_batches,
    test_connection as check_connection,
)


class FakeCursor:
    def __init__(self):
        self.execute_calls = []
        self.executemany_calls = []

    def execute(self, query, params=None):
        self.execute_calls.append((query, params))

    def executemany(self, query, params):
        self.executemany_calls.append((query, list(params)))

    def fetchone(self):
        return ("fake-version", "LOSWATCH_APP", "LOS_WH")

    def close(self):
        pass


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


class DemoCursor:
    def __init__(self, connection):
        self.connection = connection
        self.pending_count = 0

    def execute(self, query, params=None):
        table = re.search(r"LOSWATCH_[A-Z_]+", query).group(0)
        if query.startswith("SELECT COUNT"):
            self.pending_count = len(self.connection.rows.get(table, set()))

    def executemany(self, query, params):
        table = re.search(r"LOSWATCH_[A-Z_]+", query).group(0)
        self.connection.rows.setdefault(table, set()).update(
            json.dumps(row, sort_keys=True, default=str) for row in params
        )

    def fetchone(self):
        return (self.pending_count,)

    def close(self):
        pass


class DemoConnection:
    def __init__(self):
        self.rows = {}
        self.commits = 0

    def cursor(self):
        return DemoCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        pass


class FakeConnector:
    def __init__(self):
        self.kwargs = None
        self.connection = FakeConnection()

    def connect(self, **kwargs):
        self.kwargs = kwargs
        return self.connection


def configured():
    return SnowflakeConfig(
        account="example-account",
        user="operator",
        password="not-for-logs",
        warehouse="LOS_WH",
        database="LOS_DB",
        schema="LOS_SCHEMA",
    )


def test_config_reads_environment_and_safe_status_excludes_secret():
    config = SnowflakeConfig.from_env({
        "LOSWATCH_SNOWFLAKE_ACCOUNT": "account",
        "LOSWATCH_SNOWFLAKE_USER": "user",
        "LOSWATCH_SNOWFLAKE_PASSWORD": "secret-value",
        "LOSWATCH_SNOWFLAKE_WAREHOUSE": "warehouse",
        "LOSWATCH_SNOWFLAKE_DATABASE": "database",
        "LOSWATCH_SNOWFLAKE_SCHEMA": "schema",
    })
    status = config.safe_summary()
    assert status["configured"] is True
    assert "secret-value" not in json.dumps(status)
    assert "password" not in status


def test_dotenv_is_loaded_without_overriding_process_environment(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("LOSWATCH_SNOWFLAKE_ACCOUNT=from-file\nLOSWATCH_SNOWFLAKE_PASSWORD='file-secret'\n", encoding="utf-8")
    config = SnowflakeConfig.from_env(
        {"LOSWATCH_SNOWFLAKE_ACCOUNT": "from-process"}, dotenv_path=dotenv
    )
    assert config.account == "from-process"
    assert config.password == "file-secret"
    assert "file-secret" not in json.dumps(config.safe_summary())


def test_org_account_alias_supports_externalbrowser_without_password():
    config = SnowflakeConfig.from_env({
        "LOSWATCH_SNOWFLAKE_ORG_ACCOUNT": "org-account",
        "LOSWATCH_SNOWFLAKE_USER": "operator",
        "LOSWATCH_SNOWFLAKE_AUTHENTICATOR": "externalbrowser",
        "LOSWATCH_SNOWFLAKE_WAREHOUSE": "LOS_WH",
        "LOSWATCH_SNOWFLAKE_DATABASE": "LOSWATCH_AI",
        "LOSWATCH_SNOWFLAKE_SCHEMA": "MONITORING",
        "LOSWATCH_SNOWFLAKE_ROLE": "LOSWATCH_APP",
    }, dotenv_path=None)
    assert config.account == "org-account"
    assert "password" not in config.connection_kwargs()
    assert config.connection_kwargs()["authenticator"] == "externalbrowser"


def test_local_oauth_authorization_code_needs_no_password_or_client_secret():
    config = SnowflakeConfig(
        account="org-account",
        user="operator",
        warehouse="COMPUTE_WH",
        database="LOSWATCH_AI",
        schema="MONITORING",
        role="LOSWATCH_APP",
        authenticator="oauth_authorization_code",
    )
    kwargs = config.connection_kwargs()
    assert kwargs["authenticator"] == "OAUTH_AUTHORIZATION_CODE"
    assert "password" not in kwargs
    assert "oauth_client_id" not in kwargs
    assert "oauth_client_secret" not in kwargs


def test_short_snowflake_env_aliases_complete_local_configuration():
    config = SnowflakeConfig.from_env({
        "LOSWATCH_SNOWFLAKE_ACCOUNT": "org-account",
        "LOSWATCH_SNOWFLAKE_USER": "operator",
        "LOSWATCH_SNOWFLAKE_AUTHENTICATOR": "externalbrowser",
        "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
        "SNOWFLAKE_DATABASE": "LOSWATCH_AI",
        "SNOWFLAKE_SCHEMA": "MONITORING",
        "SNOWFLAKE_ROLE": "LOSWATCH_APP",
    }, dotenv_path=None)
    status = config.safe_summary()
    assert status["configured"] is True
    assert status["missing"] == []
    assert config.warehouse == "COMPUTE_WH"
    assert config.database == "LOSWATCH_AI"
    assert config.schema == "MONITORING"
    assert config.role == "LOSWATCH_APP"
    assert "password" not in config.connection_kwargs()


def test_connect_is_lazy_and_passes_credentials_only_to_connector():
    fake = FakeConnector()
    connection = connect(configured(), connector_module=fake)
    assert connection is fake.connection
    assert fake.kwargs["password"] == "not-for-logs"
    assert fake.kwargs["application"] == "LOSWatch-AI"


def test_read_only_connection_status_uses_mock_connector():
    fake = FakeConnector()
    result = check_connection(configured(), connector_module=fake)
    assert result["connected"] is True
    assert result["version"] == "fake-version"
    assert result["role"] == "LOSWATCH_APP"
    assert result["warehouse"] == "LOS_WH"


def test_connection_failure_exposes_safe_code_sqlstate_and_category_only():
    class DatabaseError(RuntimeError):
        errno = 390190
        sqlstate = "08001"

    class Connector:
        def connect(self, **kwargs):
            raise DatabaseError("390190 (08001): SAML identity provider account parameter; password=do-not-surface")

    result = check_connection(configured(), connector_module=Connector())
    assert result["connected"] is False
    assert result["error_code"] == 390190
    assert result["sqlstate"] == "08001"
    assert result["diagnostic_category"] == "saml_account_mismatch"
    assert "do-not-surface" not in json.dumps(result)
    assert "SAML identity provider account parameter" not in json.dumps(result)


def test_setup_preview_is_read_only_and_never_creates_warehouse():
    preview = setup_preview(configured())
    assert preview["read_only"] is True
    assert preview["statement_count"] == 6
    assert "no CREATE WAREHOUSE" in preview["warehouse_action"]
    assert all("CREATE WAREHOUSE" not in statement.upper() for statement in preview["statements"])


def test_admin_setup_preview_is_reviewed_and_read_only():
    preview = admin_setup_preview(configured())
    assert preview["read_only"] is True
    assert preview["database"] == "LOSWATCH_AI"
    assert preview["role"] == "LOSWATCH_APP"
    assert "CREATE WAREHOUSE" not in preview["statements"].upper()
    assert "ACCOUNTADMIN" in preview["statements"]


def test_reviewed_admin_sql_uses_existing_compute_and_documents_privilege_lifecycle():
    sql = Path(__file__).parents[1].joinpath("sql", "snowflake_admin_setup.sql").read_text(encoding="utf-8")
    assert "GRANT USAGE ON WAREHOUSE COMPUTE_WH TO ROLE LOSWATCH_APP;" in sql
    assert "LOSWATCH_EXISTING_WH" not in sql
    assert "GRANT ROLE LOSWATCH_APP TO USER <service_user>;" in sql
    assert "GRANT CREATE TABLE ON SCHEMA LOSWATCH_AI.MONITORING TO ROLE LOSWATCH_APP;" in sql
    assert "GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA LOSWATCH_AI.MONITORING TO ROLE LOSWATCH_APP;" in sql
    assert "REVOKE CREATE TABLE ON SCHEMA LOSWATCH_AI.MONITORING FROM ROLE LOSWATCH_APP;" in sql
    assert "SHOW GRANTS ON TABLE LOSWATCH_AI.MONITORING.LOSWATCH_HISTORICAL_ENCOUNTERS;" in sql
    assert "CREATE WAREHOUSE" not in sql.upper()


def test_env_example_uses_supported_names_and_is_locally_ignored():
    root = Path(__file__).parents[1]
    example = root.joinpath(".env.example").read_text(encoding="utf-8")
    ignored = root.joinpath(".gitignore").read_text(encoding="utf-8").splitlines()
    assert "LOSWATCH_SNOWFLAKE_ACCOUNT=" in example
    assert "LOSWATCH_SNOWFLAKE_ORG_ACCOUNT=" in example
    assert "LOSWATCH_SNOWFLAKE_AUTHENTICATOR=OAUTH_AUTHORIZATION_CODE" in example
    assert "oauth_client_secret" not in example.lower()
    assert "LOSWATCH_SNOWFLAKE_ROLE=LOSWATCH_APP" in example
    assert ".env" in ignored


def test_missing_optional_connector_is_actionable(monkeypatch):
    monkeypatch.setattr("loswatch.snowflake.importlib.import_module", lambda _: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(SnowflakeError, match="optional"):
        connect(configured())


def test_historical_sync_rejects_identifiers_and_accepts_sanitized_features():
    raw_like = pd.DataFrame([{**{feature: 0 for feature in FEATURES}, "patient_nbr": "123"}])
    with pytest.raises(SnowflakeValidationError, match="de-identified"):
        historical_records(raw_like)

    sanitized = pd.DataFrame([{**{feature: 0 for feature in FEATURES}, "actual_los_days": 4}])
    records = historical_records(sanitized)
    assert len(records) == 1
    assert records[0]["record_key"].startswith("row-")
    assert "patient_nbr" not in records[0]


def test_sync_uses_parameterized_idempotent_merge_and_is_repeatable():
    connection = FakeConnection()
    record = {
        "prediction_key": "prediction-abc123",
        "source_batch_id": "batch-abc123",
        "model_version": "phase21",
        "predicted_los_days": 4.25,
        "feature_fingerprint": "fingerprint",
        "created_utc": "2026-11-22T00:00:00+00:00",
    }
    first = sync_batch(connection, configured(), "prediction_logs", [record])
    second = sync_batch(connection, configured(), "prediction_logs", [record])
    assert first.rows_submitted == second.rows_submitted == 1
    assert first.batch_id == second.batch_id == "batch-abc123"
    assert len(connection.cursor_instance.executemany_calls) == 2
    query, parameters = connection.cursor_instance.executemany_calls[0]
    assert query.startswith("MERGE INTO")
    assert "%(prediction_key)s" in query
    assert parameters[0]["prediction_key"] == "prediction-abc123"
    assert connection.commits == 2


def test_schema_creation_and_preview_are_non_uploading():
    connection = FakeConnection()
    executed = ensure_schema(connection, configured())
    assert executed == 6
    assert len(connection.cursor_instance.execute_calls) == 6
    preview = preview_batch("prediction_logs", [{
        "prediction_key": "prediction-1",
        "source_batch_id": "batch-1",
        "model_version": "phase21",
        "predicted_los_days": 3.0,
        "feature_fingerprint": "fp",
        "created_utc": "2026-11-22T00:00:00+00:00",
    }])
    assert preview["upload_performed"] is False
    assert preview["rows"] == 1


def test_initialization_requires_exact_manual_confirmation():
    connection = FakeConnection()
    with pytest.raises(SnowflakeValidationError, match="exact confirmation"):
        initialize_snowflake(connection, configured(), "yes")
    result = initialize_snowflake(connection, configured(), INITIALIZATION_CONFIRMATION)
    assert result["initialized"] is True
    assert result["statement_count"] == 5
    assert result["schema_created"] is False
    assert result["warehouse_created"] is False


def test_synthetic_demo_sync_is_duplicate_safe_and_query_back_verified():
    batches = synthetic_demo_batches()
    assert set(batches) == {"historical_encounters", "prediction_logs", "drift_reports", "model_versions", "retraining_audit_events"}
    assert all("patient_nbr" not in row and "encounter_id" not in row for rows in batches.values() for row in rows)
    connection = DemoConnection()
    first = sync_synthetic_demo(connection, configured(), DEMO_SYNC_CONFIRMATION)
    second = sync_synthetic_demo(connection, configured(), DEMO_SYNC_CONFIRMATION)
    assert first["synthetic_only"] is True
    assert second["query_back_counts"] == first["query_back_counts"]
    assert all(item["duplicate_safe"] for item in second["tables"].values())
    assert all(item["stored_count"] == item["submitted"] for item in second["tables"].values())
    assert connection.commits == 10
