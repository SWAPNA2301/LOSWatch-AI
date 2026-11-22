# LOSWatch-AI — Phases 1 and 2

A local Python / Streamlit / SQLite / SQLAlchemy application for **entirely fictional** hospital operations, plus a separate offline ML research workflow. No medical recommendations, cloud services, or dashboard ML integration. The dashboard uses fictional records; Phase 2 uses the public de-identified UCI dataset only in a separate training directory.

## Quick start

Python 3.11+ recommended (tested here on Python 3.14). From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m loswatch.seed
python -m streamlit run app.py
```

Open http://localhost:8501. For macOS/Linux, activate with `source .venv/bin/activate` instead. Without activation, use `.venv\Scripts\python.exe` on Windows in place of `python`.

```powershell
python -m pytest -q
```

The seed command inserts 20 fictional patients, four fictional doctors, 24 beds, 20 diagnoses, 20 recorded medication orders, and 20 admissions (15 active, five discharged). It runs in one transaction and skips a nonempty database. Names and histories were invented; any resemblance is coincidental. Medication names are illustrative recorded data with no dosing guidance.

## Using the dashboard

- **Patients:** search by name or MRN; inspect diseases, doctor, medication orders, current bed, admission history, and elapsed calendar days of the current stay.
- **Doctors:** view each doctor's assigned patients and unassigned patients.
- **Beds:** inspect occupied and available beds with current occupants.
- **Manage records:** create, inspect, update, or delete any of the six record types. Create doctors and beds first, then patients, then dependent records. Assign doctors through Patient updates. Create an Admission to occupy a bed; update its bed to transfer; set its discharge date to release it. Clearing that date reopens an admission only if both patient and bed are free.

Dates are local calendar dates, not timestamps. A same-day admission has zero elapsed days. A null discharge date denotes an active admission. Medication end dates may be in the future; diagnoses, birth dates, and admission/discharge dates may not. Medication orders remain historical records and are not automatically ended on discharge. A patient's assigned doctor is the current assignment, not a historical assignment per admission.

## Structure

```text
app.py                  Streamlit entry point
loswatch/
  database.py           Engine, foreign-key enforcement, session factory
  models.py             Six relational models and database constraints
  services.py           Atomic CRUD, validation, search, occupancy
  seed.py               Deterministic fictional seed command
  ui.py                 Dashboard views and CRUD forms
tests/                  Service, integrity, concurrency, and UI tests
requirements.txt        Runtime and test dependencies
pytest.ini              Test discovery configuration
.streamlit/config.toml  Localhost binding and disabled usage telemetry
```

The default database is `data/loswatch.db`, independent of the launch working directory. Set `LOSWATCH_DATABASE_URL` to a SQLite URL to override it (the parent directory must exist). Tests use isolated temporary databases and do not change demo data. Schema creation runs on startup; schema migrations are outside Phase 1.

## Integrity and limits

Foreign keys are enabled on every connection. Unique partial indexes enforce one active admission per patient and per bed, including concurrent writes. MRNs and bed labels are unique. Date ordering and required text have database checks; the service layer additionally validates dates, lengths, choices, and references. Birth-date edits cannot invalidate child records. Failed writes roll back. Referenced parents cannot be deleted; explicitly remove or reassign dependencies first. Transfers update the admission's bed; a bed-transfer history and historical bed overlap checking are not part of this phase.

This is a local demonstration without authentication, authorization, audit trails, backups, or production clinical controls. Use fictional data only and run on localhost. CRUD deletion is permanent. Dependencies use bounded versions rather than a frozen lockfile.

## Phase 2: offline LOS training

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.venv\Scripts\python.exe -m loswatch.ml.train
.venv\Scripts\python.exe -m pytest -q
```

The command downloads [UCI dataset 296](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008), caches it with SHA-256 provenance in `data/uci296/`, and writes `artifacts/phase2/`. It never opens the application SQLite database. Existing output directories are refused; an intentional future experiment requires a distinct `--output` path. There is no scheduled retraining or model-serving path. Phase 2 dependencies are optional for the dashboard; ML tests skip explicitly if scikit-learn is absent.

### Target and leakage policy

Target: `time_in_hospital`, integer days from 1 to 14. UCI selected diabetic inpatient encounters that had labs and medications and lasted within this range; results are conditional on that retrospective cohort, not a general admission population.

An explicit admission-time allowlist accepts only age band, gender, admission type, admission source, and counts of outpatient/emergency/inpatient visits **in the preceding year**. Age becomes the band midpoint, and prior utilization produces a total and an inpatient-history indicator. Admission codes are categorical, not ordinal. Missing/unknown admission mapping codes become missing. Invalid ages and negative/noninteger history counts become missing rather than invented values.

Excluded: target, encounter/patient identifiers as predictors, discharge disposition, readmission, diagnoses and diagnosis counts, all encounter lab/procedure/medication fields, glucose/A1c results, medication changes, and diabetes medication status. These may only be known during or after the stay. Weight, payer and admitting specialty are conservatively omitted because their recording time is not established. Race is omitted from this initial model. Gender and age remain demographic features; no subgroup fairness validation or clinical deployment claim is made. See the [UCI variable descriptions](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008) and accompanying `IDS_mapping.csv` for source definitions.

### Cleaning, EDA, splitting and evaluation

Exact duplicate rows are removed; missing/invalid patient or encounter IDs and invalid targets are excluded with counts recorded. Conflicting duplicate encounter IDs raise an error. No rows are excluded based on discharge or readmission outcomes. Missing markers are normalized; numeric median imputation with indicators and categorical constant imputation/one-hot encoding are fitted only on training data. Unseen categories are safely ignored. Tree models need no scaling.

Two seeded GroupShuffleSplit operations produce approximately 60/20/20 percent **of patients** for train/validation/test; encounter ratios can differ. Assertions and tests enforce pairwise patient disjointness. EDA is restricted to training data and records LOS distribution, missingness, numeric summaries, category frequencies, age/LOS summaries, and repeat encounters. No patient or encounter ID is a model feature.

Models: median DummyRegressor (MAE baseline), regularized Random Forest, and HistGradientBoosting. Hyperparameters are fixed before evaluation. HGB early stopping is disabled to avoid an internal random-row split across repeated patients. Best means lowest **validation MAE**, with a deterministic tie-break. Test MAE, RMSE (both in days), and R² are reported for every model only after selection is frozen. The selected fitted training pipeline is saved without refitting. No test tuning, cross-validation claims, or temporal generalization claims are made.

### Outputs

- `best_pipeline.joblib`: selected pipeline, including admission filtering, engineering, imputation and encoding. Load only trusted local joblib artifacts; loading executes Python serialization. Keep the source package and recorded dependency versions available.
- `metrics.json`: validation/test comparison, selection rule, counts, split audit, hyperparameters, versions and dataset fingerprint.
- `feature_schema.json`: expected inputs, transformed features, missing handling and exclusion policy.
- `eda.json`: aggregate training-only exploration.
- `split_manifest.csv`: exact patient/encounter partition assignments, kept local.
- `artifact_manifest.json`: SHA-256 fingerprints of saved outputs.

Raw data and generated artifacts are gitignored. Source attribution: Clore, Cios, DeShazo & Strack (2014), *Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning Repository, [DOI 10.24432/C5230J](https://doi.org/10.24432/C5230J), CC BY 4.0. This historical, selected US cohort cannot establish performance for current hospitals or stays outside 1–14 days.

### Completed Phase 2 run (seed 42)

All 101,766 encounters passed row validation; no exact duplicates were found. The split contains 61,215 / 20,398 / 20,153 encounters from 42,910 / 14,304 / 14,304 distinct patients, respectively, with zero patient overlap. Training LOS has median 4 days, mean 4.40 days, and an interquartile range of 2–6 days. Weight is missing in 96.8% of training records; payer and specialty are missing in 39.8% and 48.9%. Those fields were excluded by the predeclared admission-time policy, not selected using these statistics.

| Model | Validation MAE | Test MAE (days) | Test RMSE (days) | Test R² |
| --- | ---: | ---: | ---: | ---: |
| DummyRegressor (median) | 2.267 | 2.254 | 2.993 | -0.017 |
| Random Forest | 2.321 | 2.308 | 2.934 | 0.022 |
| HistGradientBoosting | 2.318 | 2.303 | 2.929 | 0.026 |

The median baseline is the saved winner under the validation-MAE rule. Both learned models slightly improve squared-error metrics but fail to beat the baseline on MAE. The restricted admission-only feature set provides weak predictive signal in this experiment; no hyperparameters were changed after viewing test results. Full precision and version details are in the generated metrics artifact.

## Phase 4: data drift monitoring

The **Drift Monitoring** page compares reproducible current admission batches with the saved UCI reference snapshot using the same seven Phase 2.1 admission-time inputs. Numeric utilization features use KS, PSI, KL divergence and Wasserstein distance; categorical features use PSI, KL divergence and a chi-square test. Missingness and unseen categories are reported separately, and hypothesis-test p-values use Benjamini-Hochberg correction across features.

The page includes labeled no-drift and simulated numeric, categorical, missingness, unseen-category and combined scenarios. Each run records the model ID, UTC timestamp, sample sizes, thresholds, scenario and per-feature results under `artifacts/phase4/reports/`. Monitoring is read-only: it does not retrain or modify operational, historical or prediction data.

## Phase 5: retraining and model governance

The **Retraining** page evaluates labeled batches with MAE/RMSE and long-stay error metrics, combines performance and drift alerts into a retraining recommendation, trains candidate pipelines on eligible labeled data with patient-disjoint train/validation splits, and compares every candidate with the current model on the same validation patients. Candidates are versioned under `artifacts/phase5/`; promotion and rollback require an approver identity and are recorded in `artifacts/model_registry/audit.jsonl`. The active prediction model uses the approved registry pointer when present and otherwise falls back to the existing Phase 2.1 artifacts.

The original full CSV is not a genuine new production batch: it contains the same patients used for development and the frozen test holdout. Phase 5 therefore evaluates that file only on the saved test partition and reuses the saved train/validation/test assignments for candidate training. Simulated Phase 4 drift reports are explicitly labeled and ignored by operational retraining recommendations; only observed drift or labeled performance alerts can trigger one.

## Historical records dashboard

The **Historical records** navigation item is a separate, read-only view of UCI encounters. Search by encounter ID, historical patient ID, or diagnosis-code substring, and filter medication indicators. Results are paginated in groups of 50. Selecting an encounter shows diagnosis codes, all 23 medication indicators, admission type/source, discharge disposition, readmission outcome and **actual** length of stay. The data contains no calendar admission dates, patient names, named doctors or bed assignments; the app does not invent them. Diagnosis codes remain ICD-9-CM codes rather than inferred disease names.

The supplied files are staged at `data/raw/diabetic_data.csv` and `data/raw/IDS_mapping.csv`. To stage files on another checkout:

```powershell
.venv\Scripts\python.exe -m loswatch.history --source C:/Users/palch/Downloads/diabetic_data.csv --mapping C:/Users/palch/Downloads/IDS_mapping.csv
```

Run `python -m loswatch.history` to regenerate the audit of already-staged files. Staging is idempotent: matching files are reused, different existing files are rejected, and records are never appended or imported into SQLite. The dashboard reads only the raw snapshot, never concatenates it with the older UCI cache, and never loads a predictive model. Fictional operational patients, CRUD, doctor assignments and occupancy remain separate. Missing historical files show an explanatory message without blocking the operational dashboard. `LOSWATCH_HISTORY_DIR` can point to an alternate directory containing both CSVs.

### Identity, mappings and missing-value audit

Both supplied files are byte-for-byte identical to the Phase 2 download. Dataset SHA-256: `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97`. There are **101,766 unique encounters from 71,518 patients**, no exact duplicate encounters, and the existing training corpus has the same count and fingerprint. The existing Phase 2 cache and model artifacts are preserved; no new training was performed.

The parser treats the mapping file as **three independent sections**, recognizing section headers and blank separators and using CSV quoting for descriptions containing commas. It preserves the same numeric code's different meanings in each section. There are 8 admission-type, 30 discharge-disposition and 25 admission-source definitions. All observed codes resolve; documented unknown/unavailable codes remain explicitly labelled rather than being mistaken for valid clinical categories. Unrecognized codes would display `Unmapped code ...` and appear in the audit.

Missing markers `?` and empty cells are distinguished from literal `None` (a lab not measured), medication `No`, and mapping label `NULL`. The audit found 98,569 missing weights, 40,256 missing payer codes, 49,949 missing specialties, and 21 / 358 / 1,423 missing diagnosis codes at positions 1 / 2 / 3. It separately counts unknown gender and unavailable mapping codes. Glucose and A1c have 96,420 and 84,748 `None` indicators respectively. The original Phase 2 EDA used pandas default NA parsing and counted those lab indicators as missing; that historical artifact is retained, and the new audit distinguishes their meaning. These lab columns were never ML predictors, so this distinction does not require retraining.

The aggregate audit is saved as `artifacts/history_audit.json`. It checks the saved Phase 2 feature schema against the admission-only allowlist. Discharge disposition, readmission, actual LOS, encounter/patient IDs, encounter diagnoses, medications, labs, procedures, and other fields not established at admission remain excluded from model inputs. Tests change all historical non-allowlisted fields and verify that engineered ML inputs do not change. Historical outcomes are displayed for retrospective browsing only. The read-only model snapshot supplied with this task was inspected but not edited; no operational schema migration was needed.

## Phase 2.1: audited model improvement

```powershell
.venv\Scripts\python.exe -m loswatch.ml.phase21 develop
.venv\Scripts\python.exe -m loswatch.ml.phase21 evaluate
.venv\Scripts\python.exe -m pytest -q
```

The optional ML dependencies remain in `requirements-ml.txt`. Outputs go to a **new** `artifacts/phase21/` directory; development refuses an existing directory and evaluation refuses a previously evaluated run. Use `--output`, `--previous`, or `--csv` to choose explicit paths. This is an offline experiment: no dashboard, historical-view, SQLite or cloud changes.

### Audit and age parsing correction

The audit found that Phase 2's age parser expected comma-separated intervals such as `[50,60)`, but all UCI ages use hyphens such as `[50-60)`. Consequently all 61,215 training ages became missing. `AdmissionFeatures21` accepts both formats and validates decade boundaries. The original transformer implementation and artifacts remain intact for compatibility and reproducible before/after comparisons. Corrected ages and new engineered features belong only to the versioned Phase 2.1 pipeline.

The same seven raw admission inputs are retained. Candidate engineering adds log-transformed prior-year visit counts and totals, inpatient/emergency shares, an any-prior-visit indicator, an age/prior-inpatient interaction, age-band categories and admission-source groups. These are deterministic per-row transformations with no patient-ID features, outcome aggregates or cross-encounter reconstruction. Diagnoses, medication changes, within-stay tests/procedures, discharge, readmission and the target are excluded. Missing-value handling preserves the distinction between missing markers and literal lab `None`; unavailable admission codes remain missing. All learned imputations and category vocabularies fit on training data only.

### Development, tuning and final evaluation

The original patient-disjoint manifest is reused exactly. The loader checks unique encounter IDs, exact encounter/patient identity and patient disjointness, then skips unselected CSV rows before parsing partition features or outcomes. Development loads only training and validation partitions. The existing Phase 2 test set has been evaluated before, so the final result is a **reused holdout**, not a newly unseen or external validation cohort.

Eleven predeclared candidates comprise a median DummyRegressor, four Random Forest settings and six HistGradientBoosting settings. Corrected-age controls use the original features; other candidates test engineered features, depth/leaf regularization, learning rate and iteration count. Absolute-error objectives align tree fitting with MAE, alongside squared-error controls. Absolute-error forests use 8,000 bootstrap draws per tree to bound computation; all training rows remain eligible. No out-of-bag score selects models, and HGB internal early stopping is disabled. See the official [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) and [HistGradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) objective documentation.

Selection minimizes validation MAE, breaking ties by validation RMSE then trial name. One winner per family and the overall winner are serialized, together with a frozen selection, feature schema and checksums. Final evaluation verifies those files, the source code, dataset and prior artifacts before loading test data. It evaluates the three frozen family winners once, reports MAE/RMSE/R² and the Phase 2 comparison, and never fits or retunes. The best pipeline is not refitted on validation, so its reported validation score describes the exact saved model.

Artifacts include `search_plan.json`, `search_results.json`, `development_audit.json`, `validation_error_analysis.json`, `selection.json`, family pipelines, `best_pipeline.joblib`, `feature_schema.json`, `frozen_manifest.json`, and final `metrics.json`/`artifact_manifest.json`. The error analysis reports validation bias, MAE/RMSE by LOS band, age, gender and admission type, patient-macro MAE, 90th-percentile absolute error and the fraction within one day. LOS buckets are retrospective diagnostics only. Small demographic groups and long-stay errors require caution; this is not clinical validation or a fairness certification. No test-driven adjustments are made after final evaluation.

### Completed Phase 2.1 results (seed 42)

The original 61,215 / 20,398 / 20,153 encounter split is unchanged, with zero patient overlap. Corrected training age missingness fell from 61,215 to zero. The pre-existing remaining missing/unknown feature counts were one gender, 6,342 admission types and 4,305 admission sources; imputation and missing categories remain training-fitted. The target remains right-skewed: training median 4, mean 4.40, interquartile range 2–6 days.

| Model | Before test MAE | After test MAE | Before test RMSE | After test RMSE | Before test R² | After test R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DummyRegressor | 2.2542 | 2.2542 | 2.9933 | 2.9933 | -0.0173 | -0.0173 |
| Random Forest | 2.3076 | 2.2007 | 2.9344 | 2.9972 | 0.0224 | -0.0200 |
| HistGradientBoosting | 2.3033 | 2.2005 | 2.9292 | 3.0253 | 0.0258 | -0.0392 |

**Selected:** `rf_absolute_deeper`, an 80-tree absolute-error forest with maximum depth 12, minimum leaf size 40, all candidate features and 8,000 bootstrap draws per tree. Validation MAE was 2.2099 days versus 2.2667 for DummyRegressor and 2.2140 for the best HGB. Compared with the old selected median baseline, the new selected model lowers test MAE by 0.0534 days (2.37%), while test RMSE increases by 0.0040 days and R² declines slightly. HGB's marginally lower test MAE did not change the validation-based selection.

Correcting age alone lowered RF validation MAE from the Phase 2 value of 2.3207 to 2.3041 with the original squared-error settings. The largest further reduction came from the absolute-error objective; results do not establish that every engineered feature helps individually. No secondary test tuning was done.

Validation error analysis shows the saved forest overpredicts 1–3-day stays by 1.53 days on average and underpredicts 8–14-day stays by 6.12 days. Its overall prediction bias is -0.70 days; patient-macro MAE is 2.1761 days. The fraction within one day is 30.7%, versus 40.7% for the integer median baseline (unrounded predictions). Thus the small MAE improvement comes with material tradeoffs, and negative test R² indicates weak overall predictive performance. These artifacts are for offline research only. All 53 tests passed, including the original operational/historical tests and new leakage, age-parsing, partition-integrity, frozen-artifact and no-fit-on-final-evaluation tests.
