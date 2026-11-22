-- LOSWatch-AI Phase 7B reviewed manual setup.
-- Execute this file only through an approved security/admin change process.
-- Never use ACCOUNTADMIN for the LOSWatch application connection.
-- This reviewed setup uses the existing COMPUTE_WH warehouse; it does not create one.
-- This file is documentation for manual execution; LOSWatch never executes it.

CREATE DATABASE IF NOT EXISTS LOSWATCH_AI;
CREATE SCHEMA IF NOT EXISTS LOSWATCH_AI.MONITORING;
CREATE ROLE IF NOT EXISTS LOSWATCH_APP;

GRANT USAGE ON DATABASE LOSWATCH_AI TO ROLE LOSWATCH_APP;
GRANT USAGE ON SCHEMA LOSWATCH_AI.MONITORING TO ROLE LOSWATCH_APP;
GRANT USAGE ON WAREHOUSE COMPUTE_WH TO ROLE LOSWATCH_APP;

-- Temporary permission required only while the explicitly confirmed app action,
-- running as LOSWATCH_APP, creates the five application tables. The creating
-- role owns those tables; do not grant ownership to a broader administrative role.
GRANT CREATE TABLE ON SCHEMA LOSWATCH_AI.MONITORING TO ROLE LOSWATCH_APP;

-- Manual ownership check after initialization (read-only; do not run from LOSWatch):
-- SHOW GRANTS ON TABLE LOSWATCH_AI.MONITORING.LOSWATCH_HISTORICAL_ENCOUNTERS;

-- After the explicitly confirmed app initialization completes, run these two
-- statements manually. They are intentionally commented so this file can be
-- reviewed and applied in safe stages.
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA LOSWATCH_AI.MONITORING TO ROLE LOSWATCH_APP;
-- REVOKE CREATE TABLE ON SCHEMA LOSWATCH_AI.MONITORING FROM ROLE LOSWATCH_APP;

-- Grant the role to the intended service user through the same approved
-- security/admin process. Do not paste credentials into this repository.
-- GRANT ROLE LOSWATCH_APP TO USER <service_user>;
