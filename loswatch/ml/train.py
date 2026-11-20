"""One explicit local training run: python -m loswatch.ml.train."""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from loswatch.ml.data import DATA_DIR, FEATURES, ROOT, TARGET, clean_rows, download, sha256
from loswatch.ml.features import AdmissionFeatures, CATEGORICAL, NUMERIC


def split_patients(frame, seed=42):
    if frame.patient_nbr.nunique() < 10:
        raise ValueError("At least 10 distinct patients are required.")
    remaining, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=frame.patient_nbr))
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1).split(frame.iloc[remaining], groups=frame.iloc[remaining].patient_nbr))
    splits = {"train": frame.iloc[remaining[train]], "validation": frame.iloc[remaining[validation]], "test": frame.iloc[test]}
    groups = [set(part.patient_nbr) for part in splits.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise AssertionError("Patient overlap detected.")
    return splits


def make_pipeline(model):
    numeric = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("admission_features", AdmissionFeatures()),
        ("preprocess", ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)], remainder="drop")),
        ("model", model),
    ])


def candidates(seed=42):
    return {
        "DummyRegressor": DummyRegressor(strategy="median"),
        "RandomForest": RandomForestRegressor(n_estimators=160, max_depth=14, min_samples_leaf=15, max_features=0.8, n_jobs=2, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=10, early_stopping=False, random_state=seed),
    }


def evaluate(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def eda(frame):
    engineered = AdmissionFeatures().fit_transform(frame)
    return {
        "scope": "Training partition only; no test-guided feature selection",
        "patients": int(frame.patient_nbr.nunique()), "encounters": len(frame),
        "encounters_per_patient": frame.groupby("patient_nbr").size().describe().to_dict(),
        "target_summary_days": frame[TARGET].describe().to_dict(),
        "target_histogram": frame[TARGET].value_counts().sort_index().to_dict(),
        "raw_missing_fraction": frame.replace("?", np.nan).isna().mean().to_dict(),
        "engineered_missing_fraction": engineered.isna().mean().to_dict(),
        "numeric_summary": json.loads(engineered[NUMERIC].describe().to_json()),
        "category_counts": {name: engineered[name].fillna("Missing").value_counts().to_dict() for name in CATEGORICAL},
        "mean_los_by_age": frame.groupby("age")[TARGET].agg(["count", "mean", "median"]).to_dict(),
    }


def train(csv, output, seed=42, models=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing runs are never overwritten.")
    raw = pd.read_csv(csv, na_values=["?"], low_memory=False)
    frame, cleaning = clean_rows(raw)
    splits = split_patients(frame, seed)
    results, pipelines = {}, {}
    training = splits["train"]
    # All learned transformations fit only on training patients. Disable HGB's
    # internal random-row early stopping, which would mix patients internally.
    with threadpool_limits(limits=2):
        for name, estimator in (models if models is not None else candidates(seed)).items():
            print(f"Training {name} on {len(training)} encounters...", flush=True)
            pipeline = make_pipeline(estimator).fit(training[FEATURES], training[TARGET])
            pipelines[name] = pipeline
            validation = splits["validation"]
            results[name] = {"validation": evaluate(validation[TARGET], pipeline.predict(validation[FEATURES]))}
        best = min(results, key=lambda name: (results[name]["validation"]["MAE"], name))
        # Selection is frozen before any test prediction; no refitting or tuning.
        for name, pipeline in pipelines.items():
            test = splits["test"]
            results[name]["test"] = evaluate(test[TARGET], pipeline.predict(test[FEATURES]))
    output.mkdir(parents=True)
    metrics = {
        "target": TARGET, "units": "days", "seed": seed, "selected_model": best,
        "selection_rule": "lowest validation MAE; alphabetical tie-break; no refit",
        "models": results, "cleaning": cleaning, "dataset_sha256": sha256(csv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "split": {name: {"rows": len(part), "patients": int(part.patient_nbr.nunique())} for name, part in splits.items()},
        "patient_overlap": 0,
        "parameters": {name: pipeline.named_steps["model"].get_params() for name, pipeline in pipelines.items()},
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "eda.json", eda(training))
    schema = {
        "raw_required_features": FEATURES, "numeric_engineered": NUMERIC, "categorical": CATEGORICAL,
        "raw_types": {name: "nonnegative integer count" if name.startswith("number_") else "categorical string or admission code" for name in FEATURES},
        "missing_policy": "? and blank to missing; invalid ages/counts to missing; median numeric imputation and constant categorical Missing, fitted on train only",
        "engineering": "age interval midpoint; total prior visits; any prior inpatient visit; admission codes treated as categories",
        "unknown_category_policy": "one-hot handle_unknown=ignore",
        "excluded_columns": [name for name in raw.columns if name not in FEATURES],
        "availability": "Demographics, admission context and utilization in the preceding year only; IDs used only for splitting/cleaning.",
        "excluded_rationale": "Target, discharge and readmission are outcomes; diagnosis/lab/procedure/drug fields summarize this encounter. Weight, payer and specialty omitted due to uncertain timing; race omitted from this initial model.",
        "transformed_features": pipelines[best].named_steps["preprocess"].get_feature_names_out().tolist(),
    }
    write_json(output / "feature_schema.json", schema)
    joblib.dump(pipelines[best], output / "best_pipeline.joblib")
    # Exact partition reproducibility; identifiers remain local, outside source control.
    pd.concat([part[["encounter_id", "patient_nbr"]].assign(split=name) for name, part in splits.items()]).to_csv(output / "split_manifest.csv", index=False)
    write_json(output / "artifact_manifest.json", {p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({"selected_model": best, "models": results}, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "phase2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Specify a new --output directory.")
    train(download(args.data_dir), args.output, args.seed)


if __name__ == "__main__":
    main()
