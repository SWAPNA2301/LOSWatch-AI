"""Reusable audit, frozen patient partitions and validation-only error analysis."""
from pathlib import Path

import numpy as np
import pandas as pd

from loswatch.ml.data import FEATURES, TARGET, clean_rows, sha256
from loswatch.ml.features import AdmissionFeatures
from loswatch.ml.features21 import AdmissionFeatures21
from loswatch.ml.train import evaluate


class FrozenCorpus:
    """Check IDs first; parse features/outcomes only for the requested partition."""

    def __init__(self, csv_path, split_path, expected_hash=None):
        self.csv_path = Path(csv_path)
        self.split_path = Path(split_path)
        self.dataset_hash = sha256(self.csv_path)
        self.split_hash = sha256(self.split_path)
        if expected_hash is not None and self.dataset_hash != expected_hash:
            raise ValueError("Dataset fingerprint differs from the frozen experiment.")
        self.manifest = pd.read_csv(self.split_path, dtype=str, keep_default_na=False)
        if set(self.manifest.columns) != {"encounter_id", "patient_nbr", "split"}:
            raise ValueError("Invalid split manifest columns.")
        if set(self.manifest.split) != {"train", "validation", "test"}:
            raise ValueError("Exactly three nonempty partitions are required.")
        if self.manifest.encounter_id.duplicated().any():
            raise ValueError("Duplicate encounters in the split manifest.")
        if (self.manifest.groupby("patient_nbr").split.nunique() != 1).any():
            raise ValueError("Patient overlap across partitions.")
        self.identities = pd.read_csv(self.csv_path, usecols=["encounter_id", "patient_nbr"], dtype=str, keep_default_na=False)
        for column in ("encounter_id", "patient_nbr"):
            if not self.identities[column].str.fullmatch(r"\d+", na=False).all():
                raise ValueError("Invalid encounter/patient identifiers.")
        if self.identities.encounter_id.duplicated().any():
            raise ValueError("Duplicate encounters in the dataset.")
        expected = self.manifest.set_index("encounter_id").patient_nbr.sort_index()
        actual = self.identities.set_index("encounter_id").patient_nbr.sort_index()
        if not actual.equals(expected):
            raise ValueError("Dataset encounters/patients do not exactly match the frozen split.")

    def load(self, partition):
        if partition not in {"train", "validation", "test"}:
            raise ValueError("Unknown partition.")
        ids = set(self.manifest.loc[self.manifest.split == partition, "encounter_id"])
        positions = set(self.identities.index[self.identities.encounter_id.isin(ids)])
        raw = pd.read_csv(
            self.csv_path, keep_default_na=False, na_values=["?", ""], low_memory=False,
            skiprows=lambda line: line > 0 and line - 1 not in positions,
        )
        frame, cleaning = clean_rows(raw)
        if len(frame) != len(positions):
            raise ValueError("Cleaning would change the frozen cohort; investigate before training.")
        return frame

    def summary(self):
        return {
            name: {"rows": len(part), "patients": int(part.patient_nbr.nunique())}
            for name, part in self.manifest.groupby("split", sort=False)
        }


def development_audit(training, validation, corpus):
    old = AdmissionFeatures().fit_transform(training)
    corrected = AdmissionFeatures21().fit_transform(training)
    return {
        "scope": "Features and outcomes audited only on training/validation; test IDs checked for partition integrity.",
        "dataset_sha256": corpus.dataset_hash, "split_sha256": corpus.split_hash,
        "split": corpus.summary(), "patient_overlap": 0,
        "training_target": training[TARGET].describe().to_dict(),
        "training_target_histogram": training[TARGET].value_counts().sort_index().to_dict(),
        "validation_target": validation[TARGET].describe().to_dict(),
        "training_raw_missing_counts": training.isna().sum().to_dict(),
        "training_lab_not_measured": {name: int(training[name].eq("None").sum()) for name in ("max_glu_serum", "A1Cresult") if name in training},
        "old_feature_missing_counts": old.isna().sum().to_dict(),
        "corrected_feature_missing_counts": corrected.isna().sum().to_dict(),
        "age_parse_fix": "UCI uses hyphens; old comma-only parsing made all ages missing. New transformer accepts both syntaxes; old transformer is preserved.",
        "original_age_bands": sorted(training.age.dropna().unique().tolist()),
        "allowed_raw_inputs": FEATURES,
        "excluded_fields": [name for name in training if name not in FEATURES],
        "leakage_policy": "Only age, gender, admission type/source and prior-year utilization; no IDs, target, diagnoses, labs, drug changes, discharge, readmission or within-stay counts as predictors. No patient target aggregates. Every learned preprocessing step fits on training only.",
        "known_limitations": ["Historical selected diabetic 1–14-day cohort", "No temporal or external validation", "Test set was previously evaluated in Phase 2", "Age/gender subgroup errors are descriptive, not a fairness certification"],
    }


def error_analysis(frame, predicted):
    """Aggregate validation residuals; target buckets diagnose errors, never features."""
    actual = frame[TARGET].to_numpy(dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    residual = prediction - actual
    errors = pd.DataFrame({
        "actual": actual, "predicted": prediction, "error": residual,
        "absolute_error": np.abs(residual), "patient": frame.patient_nbr.to_numpy(),
        "age": frame.age.fillna("Missing").to_numpy(),
        "gender": frame.gender.fillna("Missing").to_numpy(),
        "admission_type": frame.admission_type_id.fillna("Missing").astype(str).to_numpy(),
        "los_band": pd.cut(actual, [0, 3, 7, 14], labels=["1–3", "4–7", "8–14"]),
    })
    grouped = {}
    for key in ("los_band", "age", "gender", "admission_type"):
        grouped[key] = []
        for label, group in errors.groupby(key, observed=True):
            grouped[key].append({
                "group": str(label), "encounters": len(group),
                "MAE": float(group.absolute_error.mean()),
                "RMSE": float(np.sqrt(np.mean(group.error**2))),
                "bias_predicted_minus_actual": float(group.error.mean()),
                "mean_actual": float(group.actual.mean()), "mean_predicted": float(group.predicted.mean()),
            })
    return {
        "scope": "validation only", "overall": evaluate(actual, prediction),
        "mean_bias": float(residual.mean()),
        "absolute_error_p90": float(np.quantile(np.abs(residual), 0.9)),
        "fraction_within_one_day": float(np.mean(np.abs(residual) <= 1)),
        "patient_macro_MAE": float(errors.groupby("patient").absolute_error.mean().mean()),
        "groups": grouped,
    }
