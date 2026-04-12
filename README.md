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
- [data/samples/ibm_presentation_test_data.csv](data/samples/ibm_presentation_test_data.csv)

Recommended order:

1. `mixed_eval_data.csv` for evaluation
2. `ibm_presentation_test_data.csv` for quick UI demonstration

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

### Top 5 Seed Results

Top 5 observed seed runs:

- `123` -> `92.00%`
- `451` -> `91.33%`
- `132` -> `88.67%`
- `143` -> `88.00%`
- `389` -> `87.33%`

Average of these top 5 seed runs:

- `89.47%` binary accuracy

Default review seed:

- `123`

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

- local Python run is recommended for live demo
- Docker is included for reproducibility and deployment packaging

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
