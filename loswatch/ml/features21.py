"""Versioned Phase 2.1 features; Phase 2 serialized transformers stay unchanged."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from loswatch.ml.data import HISTORY
from loswatch.ml.features import AdmissionFeatures, CATEGORICAL, NUMERIC

EXTRA_NUMERIC = [f"log1p_{name}" for name in HISTORY] + [
    "log1p_prior_visits_total", "prior_inpatient_share", "prior_emergency_share",
    "any_prior_visits", "age_x_prior_inpatient",
]
EXTRA_CATEGORICAL = ["admission_route", "age_band"]


class AdmissionFeatures21(TransformerMixin, BaseEstimator):
    """Only the seven original admission inputs are read; no outcome aggregates."""

    def __init__(self, engineered=True):
        self.engineered = engineered

    def fit(self, X, y=None):
        self.transform(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        output = AdmissionFeatures().transform(X)
        # Actual UCI syntax is [50-60); also retain the comma syntax accepted by
        # existing callers. Parsing is row-local and learns no cohort statistics.
        age = X.age.astype("string").str.strip().str.extract(
            r"^\[(\d+)\s*[-,]\s*(\d+)\)$"
        ).apply(pd.to_numeric, errors="coerce")
        valid = age[0].between(0, 90) & (age[0] % 10 == 0) & (age[1] - age[0] == 10)
        output["age_midpoint"] = ((age[0] + age[1]) / 2).where(valid).astype(float)
        if self.engineered:
            for name in [*HISTORY, "prior_visits_total"]:
                output[f"log1p_{name}"] = np.log1p(output[name])
            denominator = output.prior_visits_total.replace(0, 1)
            output["prior_inpatient_share"] = output.number_inpatient / denominator
            output["prior_emergency_share"] = output.number_emergency / denominator
            output["any_prior_visits"] = (output.prior_visits_total > 0).astype(float).where(output.prior_visits_total.notna())
            output["age_x_prior_inpatient"] = output.age_midpoint * output.any_prior_inpatient
            # This groups known admission sources only; discharge codes are never read.
            routes = {"1": "Referral", "2": "Referral", "3": "Referral", "7": "Emergency"}
            routes.update({str(code): "Transfer" for code in (4, 5, 6, 10, 18, 22, 25, 26)})
            output["admission_route"] = output.admission_source_id.map(
                lambda value: routes.get(value, "Other") if pd.notna(value) else np.nan
            ).astype(object)
            output["age_band"] = output.age_midpoint.map(
                lambda value: str(int(value // 10)) if pd.notna(value) else np.nan
            ).astype(object)
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        extra = EXTRA_NUMERIC + EXTRA_CATEGORICAL if self.engineered else []
        return np.array(NUMERIC + CATEGORICAL + extra, dtype=object)
