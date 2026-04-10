# Healthcare Anomaly Detection

Real-time healthcare claim risk evaluation using a hybrid anomaly detection pipeline:

- `Isolation Forest` for static claim anomalies
- `LSTM Autoencoder` for provider behavior over time
- `SHAP` for explainability
- `FastAPI` backend for inference
- `Streamlit` dashboard for single-claim and batch audit review

This repository is designed for two use cases:

1. `Project demo / instructor review`
2. `Reproducible testing on synthetic evaluation datasets derived from raw healthcare transactions`

## Author

- `Irfan Hussain`
- GitHub: [@irfu1s](https://github.com/irfu1s)

## 1. What This Project Does

The system analyzes healthcare billing claims and classifies them into:

- `Normal`
- `Suspicious`
- `High Risk`

The API also provides a binary anomaly decision:

- `NORMAL`
- `ANOMALY`

The Streamlit dashboard supports:

- single-claim analysis
- batch CSV upload and audit review
- explanation text and SHAP-based feature contribution charts

## 2. Repository Layout

```text
API/                 FastAPI backend
App/                 Streamlit frontend
data/raw/            source CSV files
data/samples/        demo and synthetic evaluation CSVs
data/processed/      processed outputs and saved evaluation results
database/            SQLite database
models/              trained model and scaler artifacts
notebooks/           EDA / notebook work (separate from app runtime)
src/                 training, scoring, inference, feature engineering
tests/               generators and terminal-based evaluation scripts
```

Important files:

- [API/api.py](API/api.py): backend API
- [App/app.py](App/app.py): Streamlit dashboard
- [src/model_training.py](src/model_training.py): model training
- [src/scoring.py](src/scoring.py): thresholding and anomaly pool generation
- [src/model_inference.py](src/model_inference.py): live inference logic
- [src/feature_engineering.py](src/feature_engineering.py): shared feature generation
- [tests/test_inference.py](tests/test_inference.py): terminal batch evaluation
- [tests/generate_mixed_eval_data.py](tests/generate_mixed_eval_data.py): calibrated demo evaluation generator
- [tests/generate_raw_mixed_eval_data.py](tests/generate_raw_mixed_eval_data.py): raw-only synthetic evaluation generator

## 3. Data and Model Files

The project expects these local assets:

- database: [database/healthcare.db](database/healthcare.db)
- trained models:
  - [models/iforest_model.pkl](models/iforest_model.pkl)
  - [models/lstm_autoencoder.keras](models/lstm_autoencoder.keras)
  - [models/scaler_static.pkl](models/scaler_static.pkl)
  - [models/scaler_seq.pkl](models/scaler_seq.pkl)
  - [models/score_stats.pkl](models/score_stats.pkl)
  - [models/threshold.pkl](models/threshold.pkl)

If these artifacts are already present, you do not need to retrain to review the project.

## 4. Python Environment

Recommended Python version:

- `Python 3.12`

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies used by the project:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `fastapi`
- `uvicorn`
- `streamlit`
- `plotly`
- `shap`

## 5. Run the Project

### Backend API

Start the FastAPI server from the project root:

```bash
uvicorn API.api:app --reload
```

Available endpoints:

- `GET /`
- `POST /evaluate_transaction`
- `POST /evaluate_batch`

Interactive docs:

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Frontend Dashboard

Start Streamlit from the project root:

```bash
streamlit run App/app.py
```

The dashboard supports:

- `Single Claim Analysis`
- `Batch Audit Dashboard`

## 6. Instructor Review Flow

If the repository already contains the database and model files, the quickest review path is:

1. install dependencies
2. start the API
3. start the Streamlit app
4. test one single claim
5. test one batch CSV

Recommended commands:

```bash
pip install -r requirements.txt
uvicorn API.api:app --reload
streamlit run App/app.py
```

### Recommended batch CSVs for review

Use one of these:

- [data/samples/raw_mixed_eval_data.csv](data/samples/raw_mixed_eval_data.csv)
- [data/samples/mixed_eval_data.csv](data/samples/mixed_eval_data.csv)
- [data/samples/ibm_presentation_test_data.csv](data/samples/ibm_presentation_test_data.csv)

Recommended order:

1. `raw_mixed_eval_data.csv` for stricter raw-derived testing
2. `mixed_eval_data.csv` for calibrated demo evaluation
3. `ibm_presentation_test_data.csv` for quick presentation testing

## 7. Terminal-Based Testing

### 7.1 Quick Database Check

```bash
python tests/test_conn.py
```

This confirms that the SQLite database can be opened and queried.

### 7.2 Run Batch Inference on an Existing CSV

```bash
python -m tests.test_inference --csv data/samples/raw_mixed_eval_data.csv
```

Output is saved to:

- [data/processed/batch_inference_results.csv](data/processed/batch_inference_results.csv)

### 7.3 Generate a Raw-Only Evaluation Dataset

This generator uses raw `Healthcare_transactions` and does not call the model during generation.

```bash
python -m tests.generate_raw_mixed_eval_data --normal-rows 75 --anomaly-rows 75 --seed 123
python -m tests.test_inference --csv data/samples/raw_mixed_eval_data.csv
```

### 7.4 Generate a Calibrated Demo Evaluation Dataset

This generator is designed for demo/stress testing and is more presentation-oriented.

```bash
python -m tests.generate_mixed_eval_data --normal-rows 75 --anomaly-rows 75 --seed 123
python -m tests.test_inference --csv data/samples/mixed_eval_data.csv
```

## 8. Rebuild Models

Only do this if you want to retrain the project.

### Step 1: Train the models

```bash
python -m src.model_training
```

This recreates:

- Isolation Forest
- LSTM autoencoder
- static scaler
- sequence scaler

### Step 2: Score the dataset and regenerate threshold/anomaly pool

```bash
python -m src.scoring
```

This recreates:

- [models/score_stats.pkl](models/score_stats.pkl)
- [models/threshold.pkl](models/threshold.pkl)
- [data/samples/flagged_fraud_transactions.csv](data/samples/flagged_fraud_transactions.csv)

## 9. Sample Outputs and Where They Go

### Sample input CSVs

Stored under:

- [data/samples](data/samples)

Examples:

- `normal_demo_data.csv`
- `anomaly_demo_data.csv`
- `mixed_eval_data.csv`
- `raw_mixed_eval_data.csv`

### Processed outputs

Stored under:

- [data/processed](data/processed)

Examples:

- `batch_inference_results.csv`
- `batch_inference_results_seed_123.csv`
- `raw_seed_sweep_summary.csv`
- `raw_seed_sweep_summary_2.csv`

## 10. Current Evaluation Baseline

### Raw-derived synthetic evaluation

Across 10 raw-derived synthetic evaluation seeds, binary accuracy ranged from `84.67%` to `92.00%`, with an average of `87.60%`. No tested seed fell below `80%`.

Seed results used for report/reference:

- `123` → `92.00%`
- `132` → `88.67%`
- `143` → `88.00%`
- `177` → `85.33%`
- `211` → `84.67%`
- `257` → `85.33%`
- `314` → `86.67%`
- `389` → `87.33%`
- `451` → `91.33%`
- `593` → `86.67%`

### Calibrated demo evaluation

The calibrated demo dataset is intended for presentation and stress testing. It is useful for demonstrating the app, but the raw-derived evaluation should be preferred for report claims.

## 11. Dashboard Behavior

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
- compact results table
- downloadable CSV report

If the uploaded CSV contains labels, the table can display:

- `True Label`
- `Predicted Label`

This makes demo validation easier during instructor review.

## 12. Notes on Notebook vs Project Runtime

The notebook work under [notebooks](notebooks) is separate from the app/runtime path.

Production/runtime components use:

- [database/healthcare.db](database/healthcare.db)
- [src](src)
- [API](API)
- [App](App)
- [tests](tests)

The notebook is not required to run the deployed system.

## 13. Recommended Review Checklist

For an instructor reviewing the repository:

1. confirm dependencies install
2. confirm database file exists
3. confirm model files exist
4. run `python tests/test_conn.py`
5. start `uvicorn API.api:app --reload`
6. start `streamlit run App/app.py`
7. test one single claim from the UI
8. upload `data/samples/raw_mixed_eval_data.csv`
9. confirm the batch results table appears
10. download the audit report CSV

## 14. Optional Docker Files

Docker files exist in the repository:

- [Dockerfile.backend](Dockerfile.backend)
- [Dockerfile.frontend](Dockerfile.frontend)
- [docker-compose.yml](docker-compose.yml)

They are optional and not required for basic review of the project. For the smoothest live demo, the local Python run is recommended because TensorFlow model loading is usually faster outside Docker on Windows.

Run with Docker:

```bash
docker compose up --build
```

Then open:

- Frontend: [http://localhost:8501](http://localhost:8501)
- Backend API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Stop the containers:

```bash
docker compose down
```

Docker notes:

- The frontend container calls the backend container using `http://backend:8000`.
- The backend still requires the trained files in `models/`.
- The backend still requires `database/healthcare.db`.
- Docker startup can be slower than local execution because TensorFlow and the saved model artifacts are loaded inside the container.

## 15. Production Deployment Notes

The current repository is structured for local review and Docker-based reproducibility. A production deployment can be moved to Azure by separating the system into two services:

- FastAPI backend deployed as a containerized API service.
- Streamlit frontend deployed as a containerized dashboard service.

Recommended Azure direction:

- Build and push the backend and frontend Docker images to Azure Container Registry.
- Deploy both containers using Azure Container Apps or Azure App Service for Containers.
- Store runtime configuration such as `BACKEND_URL` and `BATCH_BACKEND_URL` as environment variables.
- Store database/model artifacts in a managed storage location instead of relying on local machine paths.
- Add authentication before exposing the dashboard outside a trusted review/demo environment.

This Azure deployment path is a future production direction. The submitted project currently focuses on local execution, Docker reproducibility, and demonstration of the anomaly detection workflow.

## 16. Summary

This repository contains:

- a trained hybrid anomaly detection system
- a working FastAPI backend
- a working Streamlit audit dashboard
- raw-derived and calibrated evaluation workflows
- reproducible terminal-based testing

For most reviewers, the fastest meaningful check is:

```bash
pip install -r requirements.txt
uvicorn API.api:app --reload
streamlit run App/app.py
python -m tests.test_inference --csv data/samples/raw_mixed_eval_data.csv
```
