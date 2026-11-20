"""Download UCI 296 with provenance and deterministic row-level cleaning."""
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "uci296"
URL = "https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
TARGET = "time_in_hospital"
HISTORY = ["number_outpatient", "number_emergency", "number_inpatient"]
FEATURES = ["age", "gender", "admission_type_id", "admission_source_id", *HISTORY]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(directory=DATA_DIR):
    directory = Path(directory)
    csv = directory / "diabetic_data.csv"
    manifest = directory / "source.json"
    if csv.exists():
        if not manifest.exists() or json.loads(manifest.read_text())["csv_sha256"] != sha256(csv):
            raise ValueError("Cached dataset has missing or mismatched provenance.")
        return csv
    directory.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=120) as response:
        payload = response.read()
    with ZipFile(io.BytesIO(payload)) as archive:
        # Read only named files; never extract arbitrary archive paths.
        contents = {name: archive.read(name) for name in ("diabetic_data.csv", "IDS_mapping.csv")}
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    manifest.write_text(json.dumps({
        "dataset_id": 296, "url": URL, "doi": "10.24432/C5230J",
        "license": "CC BY 4.0", "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(payload).hexdigest(), "csv_sha256": sha256(csv),
    }, indent=2), encoding="utf-8")
    return csv


def clean_rows(raw):
    required = ["encounter_id", "patient_nbr", TARGET, *FEATURES]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = raw.copy()
    before = len(frame)
    frame = frame.drop_duplicates()
    duplicates = before - len(frame)
    for name in ("encounter_id", "patient_nbr", TARGET):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    valid = frame[TARGET].between(1, 14) & (frame[TARGET] % 1 == 0)
    for name in ("encounter_id", "patient_nbr"):
        valid &= frame[name].notna() & (frame[name] >= 0) & (frame[name] % 1 == 0)
    invalid = int((~valid).sum())
    frame = frame.loc[valid].copy()
    if frame.encounter_id.duplicated().any():
        raise ValueError("Conflicting rows share an encounter_id.")
    frame[["encounter_id", "patient_nbr", TARGET]] = frame[["encounter_id", "patient_nbr", TARGET]].astype("int64")
    frame = frame.reset_index(drop=True)
    return frame, {"raw_rows": before, "exact_duplicates_removed": duplicates,
                   "invalid_id_or_target_rows_removed": invalid, "clean_rows": len(frame)}
