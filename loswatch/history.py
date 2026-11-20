"""Read-only UCI historical encounters, independent of the operational database."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import pandas as pd

from loswatch.ml.data import FEATURES, ROOT, sha256

RAW_DIR = ROOT / "data" / "raw"
SECTIONS = ("admission_type_id", "discharge_disposition_id", "admission_source_id")
DIAGNOSES = ["diag_1", "diag_2", "diag_3"]
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
UNKNOWN_LABELS = {"null", "not available", "not mapped", "unknown/invalid"}


def parse_mappings(path):
    """Parse section headers, separator rows and quoted descriptions with commas."""
    mappings = {}
    section = None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != 2:
                raise ValueError(f"Mapping line {line_number} must contain two CSV fields.")
            code, description = (cell.strip() for cell in row)
            if code in SECTIONS and description.lower() == "description":
                if code in mappings:
                    raise ValueError(f"Repeated mapping section: {code}")
                section = code
                mappings[section] = {}
                continue
            if section is None or not code.isdigit() or not description:
                raise ValueError(f"Invalid mapping entry on line {line_number}.")
            if code in mappings[section]:
                raise ValueError(f"Duplicate code {code} in {section}.")
            mappings[section][code] = description
    if set(mappings) != set(SECTIONS) or any(not values for values in mappings.values()):
        raise ValueError("All three nonempty mapping sections are required.")
    return mappings


def read_encounters(path):
    # Explicit NA policy: pandas defaults wrongly collapse literal 'None' (lab
    # not measured) and 'NULL'. Raw codes remain strings, including ICD decimals.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["?", ""])
    required = {"encounter_id", "patient_nbr", "time_in_hospital", *FEATURES,
                *SECTIONS, *DIAGNOSES, *MEDICATIONS, "change", "diabetesMed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Historical CSV is missing columns: {sorted(missing)}")
    frame = frame.drop_duplicates().reset_index(drop=True)
    if frame.encounter_id.duplicated().any():
        raise ValueError("Conflicting duplicate encounter IDs; historical data not loaded.")
    for column in ("encounter_id", "patient_nbr"):
        if not frame[column].str.fullmatch(r"\d+", na=False).all():
            raise ValueError(f"Invalid or missing {column}.")
    los = pd.to_numeric(frame.time_in_hospital, errors="coerce")
    if not (los.between(1, 14) & (los % 1 == 0)).all():
        raise ValueError("Historical actual LOS must be an integer from 1 to 14 days.")
    frame["time_in_hospital"] = los.astype(int)
    return frame


def mapped_label(mapping, code):
    if pd.isna(code):
        return "Missing"
    return mapping.get(str(code), f"Unmapped code {code}")


def filter_encounters(frame, encounter="", patient="", diagnosis="", medication=None, status="Any"):
    matches = pd.Series(True, index=frame.index)
    for column, query in (("encounter_id", encounter), ("patient_nbr", patient)):
        if query.strip():
            matches &= frame[column].str.contains(query.strip(), regex=False, na=False)
    if diagnosis.strip():
        matches &= pd.concat([
            frame[column].str.contains(diagnosis.strip(), case=False, regex=False, na=False)
            for column in DIAGNOSES
        ], axis=1).any(axis=1)
    if medication is not None:
        if medication not in MEDICATIONS:
            raise ValueError("Unknown medication indicator.")
        if status == "Recorded use":
            matches &= frame[medication].isin(["Steady", "Up", "Down"])
        elif status != "Any":
            matches &= frame[medication].eq(status)
    return frame.loc[matches]


def audit(csv_path, mappings_path, phase2_dir=ROOT / "artifacts" / "phase2"):
    frame = read_encounters(csv_path)
    mappings = parse_mappings(mappings_path)
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        raw_count = sum(1 for _ in csv.reader(stream)) - 1
    report = {
        "rows": len(frame), "unique_encounters": int(frame.encounter_id.nunique()),
        "patients": int(frame.patient_nbr.nunique()), "exact_duplicates_removed": raw_count - len(frame),
        "dataset_sha256": sha256(csv_path), "mapping_sha256": sha256(mappings_path),
        "missing_policy": "Only ? and empty cells are missing; literal None (not measured), No and mapping NULL are preserved.",
        "missing_counts": {name: int(count) for name, count in frame.isna().sum().items()},
        "unknown_invalid_gender_count": int(frame.gender.eq("Unknown/Invalid").sum()),
        "lab_not_measured_counts": {name: int(frame[name].eq("None").sum()) for name in ("max_glu_serum", "A1Cresult") if name in frame},
        "actual_los": frame.time_in_hospital.describe().to_dict(),
        "mappings": {},
        "ml_allowed_inputs": FEATURES,
        "ml_excluded_columns": [name for name in frame.columns if name not in FEATURES],
        "leakage_policy": "Only demographics, admission type/source and preceding-year utilization. Discharge/readmission, actual LOS, IDs, diagnoses, lab/procedure/drug indicators and other fields not established at admission are excluded from predictors. Historical display is not prediction input.",
    }
    for name, mapping in mappings.items():
        counts = frame[name].value_counts()
        report["mappings"][name] = {
            "defined_codes": len(mapping),
            "unmapped_codes": {str(code): int(count) for code, count in counts.items() if str(code) not in mapping},
            "unknown_or_unavailable_codes": {str(code): int(count) for code, count in counts.items() if mapping.get(str(code), "").lower() in UNKNOWN_LABELS},
        }
    metrics_path = Path(phase2_dir) / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        schema = json.loads((Path(phase2_dir) / "feature_schema.json").read_text())
        report["phase2"] = {
            "same_csv_bytes": metrics["dataset_sha256"] == report["dataset_sha256"],
            "training_corpus_rows": metrics["cleaning"]["clean_rows"],
            "same_encounter_count": metrics["cleaning"]["clean_rows"] == len(frame),
            "schema_matches_allowlist": schema["raw_required_features"] == FEATURES,
            "retrained": False,
        }
    else:
        report["phase2"] = {"verification": "No existing Phase 2 artifacts available", "retrained": False}
    return report


def stage_files(csv_source, mapping_source, directory=RAW_DIR):
    """Idempotent raw snapshot staging; never append rows or write to SQLite."""
    directory = Path(directory)
    read_encounters(csv_source)
    parse_mappings(mapping_source)
    pairs = [(Path(csv_source), directory / "diabetic_data.csv"),
             (Path(mapping_source), directory / "IDS_mapping.csv")]
    for source, target in pairs:
        if target.exists() and sha256(source) != sha256(target):
            raise ValueError(f"Refusing to replace different existing data: {target}")
    directory.mkdir(parents=True, exist_ok=True)
    for source, target in pairs:
        if not target.exists():
            shutil.copyfile(source, target)
    return [target for _, target in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DIR / "diabetic_data.csv")
    parser.add_argument("--mapping", type=Path, default=RAW_DIR / "IDS_mapping.csv")
    args = parser.parse_args()
    csv_path, mapping_path = stage_files(args.source, args.mapping)
    report = audit(csv_path, mapping_path)
    destination = ROOT / "artifacts" / "history_audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "patients", "mappings", "phase2")}, indent=2))


if __name__ == "__main__":
    main()
