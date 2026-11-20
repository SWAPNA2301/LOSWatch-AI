"""Stateless admission-only allowlist and feature engineering."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from loswatch.ml.data import FEATURES, HISTORY

NUMERIC = ["age_midpoint", *HISTORY, "prior_visits_total", "any_prior_inpatient"]
CATEGORICAL = ["gender", "admission_type_id", "admission_source_id"]


class AdmissionFeatures(TransformerMixin, BaseEstimator):
    """Ignore all non-allowlisted columns, including target, IDs and outcomes."""

    def fit(self, X, y=None):
        self.transform(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or not set(FEATURES).issubset(X.columns):
            raise ValueError(f"Expected a DataFrame containing {FEATURES}")
        data = X[FEATURES].copy().replace({"?": np.nan, "Unknown/Invalid": np.nan, "": np.nan})
        output = pd.DataFrame(index=X.index)
        age = data.age.astype("string").str.extract(r"^\[(\d+),(\d+)\)$").apply(pd.to_numeric, errors="coerce")
        valid_age = (age[0] >= 0) & (age[1] <= 100) & (age[1] - age[0] == 10)
        output["age_midpoint"] = ((age[0] + age[1]) / 2).where(valid_age).astype(float)
        for name in HISTORY:
            value = pd.to_numeric(data[name], errors="coerce")
            output[name] = value.where((value >= 0) & (value % 1 == 0) & np.isfinite(value)).astype(float)
        output["prior_visits_total"] = output[HISTORY].sum(axis=1, min_count=3)
        output["any_prior_inpatient"] = (output.number_inpatient > 0).astype(float).where(output.number_inpatient.notna())
        output["gender"] = data.gender.where(data.gender.isin(["Male", "Female"]), np.nan).astype(object)
        for name, unknown in (("admission_type_id", [5, 6, 8]), ("admission_source_id", [9, 15, 17, 20, 21])):
            value = pd.to_numeric(data[name], errors="coerce")
            value = value.where(~value.isin(unknown) & (value > 0) & (value % 1 == 0))
            output[name] = value.map(lambda v: str(int(v)) if pd.notna(v) else np.nan).astype(object)
        return output[NUMERIC + CATEGORICAL]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(NUMERIC + CATEGORICAL, dtype=object)
