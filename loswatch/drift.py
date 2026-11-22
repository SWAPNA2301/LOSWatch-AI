"""Phase 4 drift monitoring for the frozen Phase 2.1 admission model."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from loswatch.ml.data import FEATURES, HISTORY, ROOT

PHASE = "4"
MODEL_ID = "phase21-rf_absolute_deeper"
REFERENCE_SOURCE = ROOT / "data" / "raw" / "diabetic_data.csv"
REPORT_DIR = ROOT / "artifacts" / "phase4" / "reports"
NUMERIC_FEATURES = tuple(HISTORY)
CATEGORICAL_FEATURES = tuple(name for name in FEATURES if name not in NUMERIC_FEATURES)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing admission features: {sorted(missing)}")
    result = frame.loc[:, FEATURES].copy()
    for name in CATEGORICAL_FEATURES:
        result[name] = result[name].replace({"": np.nan, "?": np.nan, "Unknown": np.nan})
        result[name] = result[name].astype("string").str.strip()
    for name in NUMERIC_FEATURES:
        result[name] = pd.to_numeric(result[name], errors="coerce")
        result.loc[result[name] < 0, name] = np.nan
    return result


def load_reference_data(path: str | Path = REFERENCE_SOURCE) -> pd.DataFrame:
    """Load the saved UCI snapshot and retain only model admission inputs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Saved reference data not found: {path}")
    return _clean_frame(pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["?", ""]))


def build_batches(reference: pd.DataFrame, size: int = 500, seed: int = 42, scenario: str = "no_drift") -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Create deterministic reference/current batches with an explicit scenario label."""
    if size < 2:
        raise ValueError("Batch size must be at least 2.")
    base = _clean_frame(reference).reset_index(drop=True)
    if len(base) < size:
        raise ValueError("Reference data is smaller than the requested batch.")
    ref = base.sample(size, random_state=seed).reset_index(drop=True)
    current = base.sample(size, random_state=seed + 1).reset_index(drop=True)
    labels = {
        "no_drift": "No drift: independent batch from the saved reference population",
        "numeric_shift": "Simulated drift: prior-year inpatient visits shifted upward",
        "categorical_shift": "Simulated drift: admission source distribution shifted",
        "missingness_shift": "Simulated drift: increased missing prior-year utilization",
        "unseen_category": "Simulated drift: unseen admission type category introduced",
        "combined": "Simulated drift: numeric, categorical, missingness and unseen-category shifts",
    }
    if scenario not in labels:
        raise ValueError(f"Unknown drift scenario: {scenario}")
    if scenario in ("numeric_shift", "combined"):
        current["number_inpatient"] = current["number_inpatient"] + 4
    if scenario in ("categorical_shift", "combined"):
        current["admission_source_id"] = "7"
    if scenario in ("missingness_shift", "combined"):
        current.loc[current.index % 3 == 0, "number_outpatient"] = np.nan
        current.loc[current.index % 4 == 0, "number_emergency"] = np.nan
    if scenario in ("unseen_category", "combined"):
        current.loc[current.index[: max(1, size // 10)], "admission_type_id"] = "999"
    return ref, current, labels[scenario]


def _distribution(values: pd.Series, numeric: bool, bins: int = 10, edges: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    values = values.dropna()
    if numeric:
        if values.empty:
            return np.array([1.0]), ["empty"]
        edges = np.unique(edges if edges is not None else np.quantile(values.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.array([values.min() - 0.5, values.max() + 0.5])
        counts, edges = np.histogram(values.astype(float), bins=edges)
        return counts.astype(float), [f"{edges[i]:g}..{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    categories = sorted({str(value) for value in values})
    return np.array([float((values.astype(str) == category).sum()) for category in categories]), categories


def _proportions(counts: np.ndarray, total: int) -> np.ndarray:
    return counts / total if total else np.zeros_like(counts, dtype=float)


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    epsilon = 1e-6
    ref = np.clip(reference, epsilon, None)
    cur = np.clip(current, epsilon, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _kl(reference: np.ndarray, current: np.ndarray) -> float:
    epsilon = 1e-6
    ref = np.clip(reference, epsilon, None)
    cur = np.clip(current, epsilon, None)
    return float(np.sum(ref * np.log(ref / cur)))


def _bh_adjust(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted((name, value) for name, value in pvalues.items() if value is not None and np.isfinite(value))
    adjusted: dict[str, float | None] = {name: None for name in pvalues}
    running = 1.0
    for rank, (name, value) in reversed(list(enumerate(valid, start=1))):
        running = min(running, value * len(valid) / rank)
        adjusted[name] = float(min(1.0, running))
    return adjusted


def _pvalue(reference: pd.Series, current: pd.Series, numeric: bool) -> float | None:
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) < 2 or len(cur) < 2:
        return None
    if numeric:
        return float(ks_2samp(ref.astype(float), cur.astype(float), alternative="two-sided", method="auto").pvalue)
    categories = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    if len(categories) < 2:
        return 1.0
    table = np.array([[int((ref.astype(str) == c).sum()) for c in categories], [int((cur.astype(str) == c).sum()) for c in categories]])
    return float(chi2_contingency(table, correction=False)[1])


def analyze(reference: pd.DataFrame, current: pd.DataFrame, *, model_id: str = MODEL_ID, alpha: float = 0.05, thresholds: dict[str, float] | None = None, scenario: str = "observed") -> dict[str, Any]:
    """Compare two batches and return a JSON-serializable monitoring report."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    ref = _clean_frame(reference)
    cur = _clean_frame(current)
    thresholds = {"psi": 0.20, "kl": 0.10, "wasserstein": 1.0, "missing_rate": 0.05, **(thresholds or {})}
    features: dict[str, Any] = {}
    raw_pvalues: dict[str, float | None] = {}
    for name in FEATURES:
        numeric = name in NUMERIC_FEATURES
        numeric_edges = None
        if numeric:
            reference_values = ref[name].dropna().astype(float)
            if not reference_values.empty:
                quantiles = np.unique(np.quantile(reference_values, np.linspace(0, 1, 11)))
                numeric_edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf])) if len(quantiles) > 1 else np.array([-np.inf, np.inf])
        ref_values, ref_labels = _distribution(ref[name], numeric, edges=numeric_edges)
        cur_values, cur_labels = _distribution(cur[name], numeric, edges=numeric_edges)
        labels = sorted(set(ref_labels) | set(cur_labels))
        ref_counts = np.array([ref_values[ref_labels.index(label)] if label in ref_labels else 0.0 for label in labels])
        cur_counts = np.array([cur_values[cur_labels.index(label)] if label in cur_labels else 0.0 for label in labels])
        ref_prob = _proportions(ref_counts, int(ref[name].notna().sum()))
        cur_prob = _proportions(cur_counts, int(cur[name].notna().sum()))
        missing_ref = float(ref[name].isna().mean())
        missing_cur = float(cur[name].isna().mean())
        unseen = sorted(set(cur[name].dropna().astype(str)) - set(ref[name].dropna().astype(str))) if not numeric else []
        pvalue = _pvalue(ref[name], cur[name], numeric)
        raw_pvalues[name] = pvalue
        item: dict[str, Any] = {"type": "numeric" if numeric else "categorical", "sample_sizes": {"reference": len(ref), "current": len(cur)}, "psi": _psi(ref_prob, cur_prob), "kl_divergence": _kl(ref_prob, cur_prob), "ks_pvalue": pvalue if numeric else None, "categorical_pvalue": pvalue if not numeric else None, "wasserstein": float(wasserstein_distance(ref[name].dropna(), cur[name].dropna())) if numeric and ref[name].notna().any() and cur[name].notna().any() else None, "missing_rate": {"reference": missing_ref, "current": missing_cur, "delta": missing_cur - missing_ref}, "unseen_categories": unseen}
        features[name] = item
    adjusted = _bh_adjust(raw_pvalues)
    alerts = []
    for name, item in features.items():
        item["adjusted_pvalue"] = adjusted[name]
        reasons = []
        if adjusted[name] is not None and adjusted[name] <= alpha:
            reasons.append("multiple-testing-adjusted hypothesis test")
        if item["psi"] >= thresholds["psi"]:
            reasons.append("PSI threshold")
        if item["kl_divergence"] >= thresholds["kl"]:
            reasons.append("KL threshold")
        if item["wasserstein"] is not None and item["wasserstein"] >= thresholds["wasserstein"]:
            reasons.append("Wasserstein threshold")
        if abs(item["missing_rate"]["delta"]) >= thresholds["missing_rate"]:
            reasons.append("missingness shift")
        if item["unseen_categories"]:
            reasons.append("unseen category")
        item["drift"] = bool(reasons)
        item["alert_reasons"] = reasons
        if reasons:
            alerts.append({"feature": name, "reasons": reasons})
    return {"schema_version": "phase4.v1", "phase": PHASE, "model_id": model_id, "created_utc": datetime.now(timezone.utc).isoformat(), "scenario": scenario, "simulated": str(scenario).startswith("Simulated drift") or str(scenario).startswith("No drift"), "sample_sizes": {"reference": len(ref), "current": len(cur)}, "alpha": alpha, "thresholds": thresholds, "alerts": alerts, "drift_detected": bool(alerts), "features": features}


def save_report(report: dict[str, Any], directory: str | Path = REPORT_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"drift_{stamp}_{report['model_id'].replace('/', '-')}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path
