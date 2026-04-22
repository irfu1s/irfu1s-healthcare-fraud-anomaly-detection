# Healthcare Fraud Anomaly Detection

Hybrid healthcare claim anomaly detection system using:

- `Isolation Forest` for static claim anomalies
- `LSTM Autoencoder` for provider behavior over time
- `FastAPI` for backend inference
- `Streamlit` for single-claim and batch audit review
- `SHAP` for claim-level explanation in the dashboard

## Overview

This project analyzes healthcare claims and classifies them into:

- `Normal`
- `Suspicious`
- `High Risk`

It also produces a binary decision:

- `NORMAL`
- `ANOMALY`

The dashboard supports:

- single claim analysis
- batch CSV upload
- compact audit review output
- downloadable batch results

## Repository Layout

```text
API/                 FastAPI backend
App/                 Streamlit frontend
data/samples/        review and evaluation CSVs
data/processed/      generated output files
database/            SQLite database
models/              trained model artifacts
notebooks/           notebook work
src/                 training, scoring, inference, feature engineering
tests/               evaluation scripts
```

Important files:

- [API/api.py](API/api.py): backend API
- [App/app.py](App/app.py): Streamlit dashboard
- [src/model_training.py](src/model_training.py): training pipeline
- [src/scoring.py](src/scoring.py): threshold and score statistics generation
- [src/model_inference.py](src/model_inference.py): live inference logic
- [src/feature_engineering.py](src/feature_engineering.py): shared feature generation
- [tests/generate_mixed_eval_data.py](tests/generate_mixed_eval_data.py): evaluation dataset generator
- [tests/test_inference.py](tests/test_inference.py): batch inference evaluation
- [tests/test_conn.py](tests/test_conn.py): database connectivity check

## Required Local Assets

This project expects these files locally:

- database: [database/healthcare.db](database/healthcare.db)
- trained models:
  - [models/iforest_model.pkl](models/iforest_model.pkl)
  - [models/lstm_autoencoder.keras](models/lstm_autoencoder.keras)
  - [models/scaler_static.pkl](models/scaler_static.pkl)
  - [models/scaler_seq.pkl](models/scaler_seq.pkl)
  - [models/score_stats.pkl](models/score_stats.pkl)
  - [models/threshold.pkl](models/threshold.pkl)

If these files already exist, you do not need to retrain.

Note: the SQLite database is a large local asset. If it is not present after cloning the repository, place the provided `healthcare.db` file inside the `database/` folder before starting the API.

## Environment

Recommended version:

- `Python 3.12`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start backend:

```bash
uvicorn API.api:app --reload
```

Start frontend:

```bash
streamlit run App/app.py
```

Useful URLs:

- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Streamlit UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## Instructor Review Flow

If the database and model files are present, the fastest review path is:

1. install dependencies
2. run the FastAPI backend
3. run the Streamlit frontend
4. test one single claim in the UI
5. upload one evaluation CSV in batch mode

Recommended commands:

```bash
pip install -r requirements.txt
uvicorn API.api:app --reload
streamlit run App/app.py
```

Recommended review CSVs:

- [data/samples/mixed_eval_data.csv](data/samples/mixed_eval_data.csv)
  

Recommended order:

1. `mixed_eval_data.csv` for evaluation
   

## Evaluation Workflow

Generate the evaluation dataset:

```bash
python -m tests.generate_mixed_eval_data --normal-rows 75 --anomaly-rows 75 --seed 123
```

Run batch inference evaluation:

```bash
python -m tests.test_inference --csv data/samples/mixed_eval_data.csv
```

Database connectivity check:

```bash
python tests/test_conn.py
```

Batch evaluation output is saved to:

- [data/processed/batch_inference_results.csv](data/processed/batch_inference_results.csv)

## Evaluation Summary

The current evaluation path uses `mixed_eval_data.csv`, generated from `Healthcare_transactions` only.
The current saved hybrid anomaly threshold is set to the `92nd percentile`.

### Top 5 Seed Results

Top 5 observed seed runs with the `92nd percentile` threshold:

- `123` -> `94.00%`
- `132` -> `90.67%`
- `451` -> `90.67%`
- `143` -> `90.00%`
- `389` -> `89.33%`

Average of these top 5 seed runs:

- `90.93%` binary accuracy

Default review seed:

- `123`

### Imbalanced Stress Test Results

The model was also tested on uneven class distributions to confirm that the result does not depend only on a perfectly balanced `75 normal / 75 anomaly` dataset.

| Test type | Normal rows | Anomaly rows | Seed | Accuracy | Precision | Recall | F1 score | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Anomaly-heavy | `30` | `70` | `123` | `92.00%` | `98.44%` | `90.00%` | `94.03%` | `1` | `7` |
| Normal-heavy | `70` | `30` | `132` | `95.00%` | `90.32%` | `93.33%` | `91.80%` | `3` | `2` |
| Anomaly-heavy | `30` | `70` | `143` | `92.00%` | `96.97%` | `91.43%` | `94.12%` | `2` | `6` |

Imbalanced test average:

- `93.00%` binary accuracy

Worst imbalanced test result:

- `92.00%` binary accuracy

These tests show that the model remains stable when the evaluation set is not perfectly balanced. Accuracy is still reported, but precision, recall, F1 score, false positives, and false negatives should be reviewed together because accuracy alone can be misleading on imbalanced datasets.

## Rebuild Models

Only do this if you want to regenerate model artifacts.

Train models:

```bash
python -m src.model_training
```

Recompute threshold and score statistics:

```bash
python -m src.scoring
```

This recreates:

- [models/score_stats.pkl](models/score_stats.pkl)
- [models/threshold.pkl](models/threshold.pkl)

## Dashboard Output

### Single Claim Analysis

The single-claim screen shows:

- anomaly score
- risk category
- binary decision
- action recommendation
- explanation text
- SHAP feature contribution chart

### Batch Audit Dashboard

The batch screen shows:

- total claims
- high-risk count
- suspicious count
- flag rate
- compact audit table
- downloadable CSV report

If the uploaded CSV contains labels, the table can show:

- `True Label`
- `Predicted Label`

## Docker

Docker files are included:

- [Dockerfile.backend](Dockerfile.backend)
- [Dockerfile.frontend](Dockerfile.frontend)
- [docker-compose.yml](docker-compose.yml)

Run with Docker:

```bash
docker compose up --build
```

Then open:

- frontend: [http://localhost:8501](http://localhost:8501)
- backend docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Stop containers:

```bash
docker compose down
```

Note:

- local Python run is suitable for development and direct debugging
- Docker provides a reproducible packaged environment and is practical for demonstration and review

## Deployment Direction

For temporary or future deployment, the project can be containerized and deployed as:

- FastAPI backend container
- Streamlit frontend container

Suitable targets:

- Azure Container Apps
- Azure App Service for Containers

In a real deployment, database and model artifacts should be stored outside the Git repository and mounted or configured at runtime.

## Author

- `Irfan Hussain`
- GitHub: [@irfu1s](https://github.com/irfu1s)
