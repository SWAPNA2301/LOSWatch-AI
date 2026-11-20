"""Phase 2.1: separate development and final-evaluation commands; no serving."""
import argparse
import json
import platform
import shutil
import time
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from loswatch.ml.data import FEATURES, ROOT, TARGET, sha256
from loswatch.ml.experiment21 import FrozenCorpus, development_audit, error_analysis
from loswatch.ml.features import CATEGORICAL, NUMERIC
from loswatch.ml.features21 import AdmissionFeatures21, EXTRA_CATEGORICAL, EXTRA_NUMERIC
from loswatch.ml.train import evaluate, write_json

DEFAULT_DATA = ROOT / "data" / "raw" / "diabetic_data.csv"
DEFAULT_PREVIOUS = ROOT / "artifacts" / "phase2"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase21"
SELECTION_RULE = "Lowest validation MAE, then validation RMSE, then trial name; no refit after selection."


def trial_specs():
    """Fixed, small search declared before evaluation; no test-informed adaptation."""
    specs = [{"name": "dummy_median", "family": "DummyRegressor", "engineered": False, "params": {"strategy": "median"}}]
    for name, engineered, params in [
        ("rf_corrected_squared", False, dict(n_estimators=160, max_depth=14, min_samples_leaf=15, max_features=0.8)),
        ("rf_engineered_squared", True, dict(n_estimators=160, max_depth=10, min_samples_leaf=60, max_features=0.8, max_samples=0.7)),
        ("rf_absolute_shallow", True, dict(n_estimators=80, max_depth=8, min_samples_leaf=30, max_features=0.8, max_samples=8000, criterion="absolute_error")),
        ("rf_absolute_deeper", True, dict(n_estimators=80, max_depth=12, min_samples_leaf=40, max_features=1.0, max_samples=8000, criterion="absolute_error")),
    ]:
        specs.append({"name": name, "family": "RandomForest", "engineered": engineered, "params": params})
    for name, engineered, params in [
        ("hgb_corrected_squared", False, dict(loss="squared_error", max_iter=180, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=10)),
        ("hgb_corrected_absolute", False, dict(loss="absolute_error", max_iter=180, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=10)),
        ("hgb_engineered_squared", True, dict(loss="squared_error", max_iter=220, max_leaf_nodes=15, learning_rate=0.05, min_samples_leaf=60, l2_regularization=20)),
        ("hgb_absolute_small", True, dict(loss="absolute_error", max_iter=160, max_leaf_nodes=7, learning_rate=0.06, min_samples_leaf=60, l2_regularization=10)),
        ("hgb_absolute_medium", True, dict(loss="absolute_error", max_iter=250, max_leaf_nodes=15, learning_rate=0.05, min_samples_leaf=40, l2_regularization=10)),
        ("hgb_absolute_large", True, dict(loss="absolute_error", max_iter=220, max_leaf_nodes=31, learning_rate=0.05, min_samples_leaf=80, l2_regularization=20)),
    ]:
        specs.append({"name": name, "family": "HistGradientBoosting", "engineered": engineered, "params": params})
    return specs


def build_pipeline(spec, seed=42):
    family, params = spec["family"], dict(spec["params"])
    if family == "DummyRegressor":
        estimator = DummyRegressor(**params)
    elif family == "RandomForest":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=2, **params)
    elif family == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **params)
    else:
        raise ValueError(f"Unknown model family: {family}")
    numeric = NUMERIC + (EXTRA_NUMERIC if spec["engineered"] else [])
    categorical = CATEGORICAL + (EXTRA_CATEGORICAL if spec["engineered"] else [])
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")
    return Pipeline([
        ("admission_features", AdmissionFeatures21(engineered=spec["engineered"])),
        ("preprocess", preprocessing), ("model", estimator),
    ])


def score_key(result):
    return result["validation"]["MAE"], result["validation"]["RMSE"], result["name"]


def freeze_manifest(directory):
    return {path.name: sha256(path) for path in directory.iterdir() if path.is_file()}


def develop(csv_path=DEFAULT_DATA, previous=DEFAULT_PREVIOUS, output=DEFAULT_OUTPUT, seed=42, specs=None):
    """Train/validate and freeze winners without loading the test partition."""
    previous, output = Path(previous), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; prior experiments are never overwritten.")
    corpus = FrozenCorpus(csv_path, previous / "split_manifest.csv")
    training, validation = corpus.load("train"), corpus.load("validation")
    specs = trial_specs() if specs is None else specs
    if {spec["family"] for spec in specs} != {"DummyRegressor", "RandomForest", "HistGradientBoosting"}:
        raise ValueError("The baseline and both model families are required.")
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("Trial names must be unique.")
    output.mkdir(parents=True)
    plan = {
        "seed": seed, "dataset_sha256": corpus.dataset_hash,
        "split_manifest_sha256": corpus.split_hash, "split": corpus.summary(),
        "selection_rule": SELECTION_RULE, "specs": specs,
        "test_policy": "Only IDs checked during development. Final command evaluates once after winners are frozen. This is a reused Phase 2 holdout.",
        "training_policy": "All candidates see training only; numeric imputation and categorical vocabulary fitted inside each pipeline. No HGB internal split. RF bootstrap size capped for absolute-error computation; no OOB selection.",
        "previous_artifact_hashes": freeze_manifest(previous),
        "code_hashes": {name: sha256(Path(__file__).with_name(name)) for name in ("phase21.py", "experiment21.py", "features21.py", "features.py", "data.py", "train.py")},
    }
    write_json(output / "search_plan.json", plan)
    write_json(output / "development_audit.json", development_audit(training, validation, corpus))
    results, winners, errors = [], {}, {}
    with threadpool_limits(limits=2):
        for spec in specs:
            print(f"Training {spec['name']} ({len(results) + 1}/{len(specs)})...", flush=True)
            started = time.monotonic()
            pipeline = build_pipeline(spec, seed).fit(training[FEATURES], training[TARGET])
            prediction = pipeline.predict(validation[FEATURES])
            result = {
                **spec, "validation": evaluate(validation[TARGET], prediction),
                "train": evaluate(training[TARGET], pipeline.predict(training[FEATURES])),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "estimator_parameters": pipeline.named_steps["model"].get_params(),
            }
            results.append(result)
            family = spec["family"]
            if family not in winners or score_key(result) < score_key(winners[family]):
                winners[family] = result
                joblib.dump(pipeline, output / f"{family}.joblib")
                errors[family] = error_analysis(validation, prediction)
            write_json(output / "search_results.json", results)
            print(f"Validation MAE={result['validation']['MAE']:.5f}; RMSE={result['validation']['RMSE']:.5f}", flush=True)
    selected = min(winners, key=lambda family: score_key(winners[family]))
    shutil.copyfile(output / f"{selected}.joblib", output / "best_pipeline.joblib")
    pipeline = joblib.load(output / "best_pipeline.joblib")
    selection = {
        "selected_model": selected, "selected_trial": winners[selected]["name"],
        "selection_rule": SELECTION_RULE, "winners": winners,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": False,
    }
    write_json(output / "selection.json", selection)
    write_json(output / "validation_error_analysis.json", errors)
    write_json(output / "feature_schema.json", {
        "version": "2.1", "raw_required_features": FEATURES,
        "raw_types": {"age": "10-year interval [50-60) or [50,60)", "gender": "Male/Female or missing", "admission_type_id": "categorical code", "admission_source_id": "categorical code", "number_outpatient": "nonnegative integer", "number_emergency": "nonnegative integer", "number_inpatient": "nonnegative integer"},
        "selected_engineered": winners[selected]["engineered"],
        "all_candidate_engineering": "Correct hyphen/comma age-band parsing; log1p prior counts/total, prior inpatient/emergency shares, any utilization, age × prior inpatient flag, admission-source group and age-band category.",
        "engineered_features": pipeline.named_steps["admission_features"].get_feature_names_out().tolist(),
        "transformed_features": pipeline.named_steps["preprocess"].get_feature_names_out().tolist(),
        "missing_policy": "? and blank to missing; invalid ages and negative/noninteger counts to missing; documented unavailable admission codes to missing; train-only median/constant imputation with numeric indicators; unknown categories ignored.",
        "excluded_columns": [name for name in training if name not in FEATURES],
        "timing_policy": "Same seven admission-time raw inputs as Phase 2. No outcome, discharge, lab, drug, diagnosis or identifier predictors. No patient-history construction from this dataset's other encounters.",
        "prediction_units": "days; no rounding, clipping or post-hoc calibration",
    })
    write_json(output / "frozen_manifest.json", freeze_manifest(output))
    print(f"Frozen winner: {selected} / {winners[selected]['name']}. Test partition not loaded.", flush=True)
    return selection


def final_evaluate(csv_path=DEFAULT_DATA, previous=DEFAULT_PREVIOUS, output=DEFAULT_OUTPUT):
    """Evaluate frozen family winners once; cannot tune or fit in this command."""
    previous, output = Path(previous), Path(output)
    if (output / "metrics.json").exists():
        raise FileExistsError("Final evaluation already exists; it cannot be repeated in this run.")
    frozen = json.loads((output / "frozen_manifest.json").read_text())
    for name, expected in frozen.items():
        if sha256(output / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    plan = json.loads((output / "search_plan.json").read_text())
    for name, expected in plan["code_hashes"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Experiment code changed after selection: {name}")
    for name, expected in plan["previous_artifact_hashes"].items():
        if sha256(previous / name) != expected:
            raise ValueError(f"Phase 2 artifact changed: {name}")
    if sha256(previous / "split_manifest.csv") != plan["split_manifest_sha256"]:
        raise ValueError("Frozen split changed.")
    selection = json.loads((output / "selection.json").read_text())
    corpus = FrozenCorpus(csv_path, previous / "split_manifest.csv", plan["dataset_sha256"])
    test = corpus.load("test")
    results = {}
    with threadpool_limits(limits=2):
        for family, winner in selection["winners"].items():
            pipeline = joblib.load(output / f"{family}.joblib")
            predicted = pipeline.predict(test[FEATURES])
            if not np.isfinite(predicted).all():
                raise ValueError("Nonfinite final predictions.")
            results[family] = {"trial": winner["name"], "validation": winner["validation"], "test": evaluate(test[TARGET], predicted)}
    # Read Phase 2 test scores only now, after tuning and final prediction.
    before = json.loads((previous / "metrics.json").read_text())
    if before["dataset_sha256"] != corpus.dataset_hash:
        raise ValueError("Phase 2 comparison uses a different dataset.")
    metrics = {
        "phase": "2.1", "selected_model": selection["selected_model"],
        "selected_trial": selection["selected_trial"], "selection_rule": SELECTION_RULE,
        "dataset_sha256": corpus.dataset_hash, "split_manifest_sha256": corpus.split_hash,
        "split": corpus.summary(), "patient_overlap": 0, "seed": plan["seed"],
        "models": results, "before_phase2": {"selected_model": before["selected_model"], "models": before["models"]},
        "test_evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "test_policy": "One evaluation of frozen family winners after validation selection; reused Phase 2 holdout, not a fresh external estimate.",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "artifact_manifest.json", freeze_manifest(output))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "evaluate"])
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.stage == "develop":
        develop(args.csv, args.previous, args.output, args.seed)
    else:
        final_evaluate(args.csv, args.previous, args.output)


if __name__ == "__main__":
    main()
