# DataOpti AI backend

This is the existing Flask backend extended with reusable analysis services. Start Flask from the `backend` directory so `dataopti` imports resolve.

## Modules

- `app.py` keeps the Flask app and the current upload/preparation endpoints.
- `dataopti/detection.py` recognizes tabular, sensor, image, document, notebook, and archive inputs.
- `dataopti/profiling.py` builds per-column statistics and automatically infers a likely target.
- `dataopti/quality.py` calculates an explainable score from measured missing values, duplicates, outliers, invalid values, constant columns, and high-missing columns.
- `dataopti/validation.py` flags obvious numeric mistakes using column semantics (for example impossible ages and out-of-range coordinates).
- `dataopti/cleaning.py` removes duplicates, treats invalid and missing values, drops unusable columns, and prepares a separate ML-ready copy.
- `dataopti/feature_engineering.py` derives date parts and three-axis sensor magnitudes, and excludes identifier columns from model inputs.
- `dataopti/feature_selection.py` provides embedded and mutual-information selectors that are fitted inside the training pipeline.
- `dataopti/pipeline_generator.py` constructs the preprocessing and model pipelines, including keep/remove/cap IQR outlier strategies.
- `dataopti/modeling.py` performs a reproducible holdout split, trains and compares pipelines, reports selected features and importance, and creates a feature-selected CSV.
- `dataopti/drift.py` compares reference and current dataset versions with PSI for numeric columns and total variation distance for categorical columns.
- `dataopti/anomaly.py` uses Isolation Forest to flag unusual rows in numeric data; flagged rows are reported and kept.
- `tests/test_analysis_services.py` covers profile, cleaning, automatic target inference, ML training, feature-selected export, upload, and drift endpoints.
- `examples/customers.csv` is a small mixed-quality classification example; `examples/sensor_activity.csv` exercises timestamp and accelerometer feature engineering.

## Install and run on Windows PowerShell

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

The API listens on `http://127.0.0.1:5000`. In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

## Test

From the `backend` directory:

```powershell
python -m unittest discover -s tests -v
```

The React production bundle can be checked from `frontend` with `npm run build`.

## Data flow

The React client uploads files to `POST /api/upload`. Flask detects file types and returns actual profiles. `POST /api/prepare` re-reads supported tables, infers a likely target without a target picker, detects numeric anomalies, cleans a separate copy, trains three holdout-evaluated pipelines, and returns a ZIP containing prepared and feature-selected CSVs where available. When two or more tabular files are uploaded together, it automatically compares the first table with each subsequent table before cleaning and again after cleaning and ML preprocessing. The detailed JSON report is returned in the `X-DataOpti-Report` response header and displayed by the frontend. `POST /api/dynamic/analyze` remains available for integrations that send explicit reference and current versions.

Model preprocessing and feature selection are fit on training rows only. The held-out test rows are used for scoring. Training uses a fixed random seed. Small or unusable datasets get a reason in the report rather than a fabricated score. The data drift endpoint compares distributions; it does not schedule ingestion or retrain a saved model.
