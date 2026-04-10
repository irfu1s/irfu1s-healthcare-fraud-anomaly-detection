import os
# Suppress the annoying TensorFlow warning immediately
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from src.data_loader import load_provider_history
from src.feature_engineering import (
    DEFAULT_COLUMN_VALUES,
    SEQUENCE_FEATURE_COLUMNS,
    STATIC_FEATURE_COLUMNS,
    build_features,
)
from src.hybrid_scoring import compute_hybrid_score
from src.shap_explainability import (
    explain_transaction,
    explain_in_words,
    get_driver_bullets_from_shap_data,
    get_shap_data_for_plotly,
)

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
)

# ---------- GLOBALLY LOAD FROZEN MODELS & STATS ----------
print("Loading frozen models and baseline statistics for inference...")
iforest = joblib.load('models/iforest_model.pkl')
scaler_static = joblib.load('models/scaler_static.pkl')
lstm = load_model('models/lstm_autoencoder.keras')
scaler_seq = joblib.load('models/scaler_seq.pkl')
score_stats = joblib.load('models/score_stats.pkl')
GLOBAL_THRESHOLD = joblib.load('models/threshold.pkl')

static_cols = STATIC_FEATURE_COLUMNS
seq_cols = SEQUENCE_FEATURE_COLUMNS
NORMAL_ZONE_MAX = 1.30
HIGH_RISK_ZONE_MIN = 1.85


def _normalize_transaction_payload(new_transaction):
    normalized = DEFAULT_COLUMN_VALUES.copy()
    normalized.update(new_transaction)

    if normalized.get('patient_age') is not None and normalized.get('age') in (None, '', 0):
        normalized['age'] = normalized['patient_age']

    claim_type = str(normalized.get('claim_type') or 'UNKNOWN').strip()
    if 'inpatient' in claim_type.lower():
        normalized['claim_type'] = 'Inpatient'
    elif 'outpatient' in claim_type.lower():
        normalized['claim_type'] = 'Outpatient'
    else:
        normalized['claim_type'] = claim_type or 'UNKNOWN'

    normalized['service_end'] = normalized.get('service_end') or normalized.get('service_start')
    normalized['admission_date'] = normalized.get('admission_date') or normalized.get('service_start')
    normalized['primary_procedure'] = normalized.get('primary_procedure') or 'NONE'
    normalized['diagnosis_context'] = normalized.get('diagnosis_context') or 'UNKNOWN'
    normalized['provider_id'] = str(normalized.get('provider_id', 'UNKNOWN')).strip().upper()
    normalized['patient_id'] = str(normalized.get('patient_id', 'UNKNOWN')).strip().upper()

    return normalized


def classify_alert_zone(anomaly_score):
    if anomaly_score >= HIGH_RISK_ZONE_MIN:
        return "High Risk"
    if anomaly_score > NORMAL_ZONE_MAX:
        return "Suspicious"
    return "Normal"

def run_inference(new_transaction, include_explainability=True):
    new_transaction = _normalize_transaction_payload(new_transaction)
    provider_id = str(new_transaction['provider_id'])
    
    # ---------------------------------------------------------
    # 1. TEMPORAL PRE-PROCESSING
    # ---------------------------------------------------------
    start_dt = pd.to_datetime(new_transaction['service_start'], errors='coerce')
    end_dt = pd.to_datetime(new_transaction['service_end'], errors='coerce')
    duration_days = (end_dt - start_dt).days if pd.notna(start_dt) and pd.notna(end_dt) else 0
    new_transaction['service_duration'] = max(0, duration_days)
    
    # 2. Load raw provider history and append the incoming row as the scoring target.
    history = load_provider_history(provider_id)
    target_transaction = new_transaction.copy()
    target_transaction['_target_row'] = 1
    
    if not history.empty:
        history = history.copy()
        history['_target_row'] = 0
        # Rebuild from records to avoid pandas' deprecated concat/row-insert path
        # when the provider history contains columns that are entirely empty.
        history_records = history.where(pd.notna(history), None).to_dict(orient='records')
        history_records.append(target_transaction)
        history = pd.DataFrame.from_records(history_records)
    else:
        history = pd.DataFrame([target_transaction])
        
    history['service_start'] = pd.to_datetime(history['service_start'], errors='coerce')
    history = history.sort_values(['service_start', '_target_row']).reset_index(drop=True)

    target_rows = history.index[history['_target_row'].eq(1)]
    if len(target_rows) == 0:
        raise ValueError("Unable to locate the incoming transaction in provider history.")

    # Keep only historical rows up to the incoming claim, so future provider rows
    # never influence live scoring or the LSTM sequence.
    target_index_before_features = int(target_rows[-1])
    history = history.loc[:target_index_before_features].copy()
    
    # 3. Build features (This now applies the Log Transform & Hour Gap automatically)
    history = build_features(history)
    
    # Extract the exact uploaded row, not simply the provider's latest historical row.
    target_rows = history.index[history['_target_row'].eq(1)]
    if len(target_rows) == 0:
        raise ValueError("Unable to locate the incoming transaction after feature engineering.")

    target_index = int(target_rows[-1])
    latest = history.loc[target_index].copy()
    history_for_sequence = history.loc[:target_index].copy()

    # ---------------------------------------------------------
    # 4. FROZEN INFERENCE SCORING
    # ---------------------------------------------------------
    
    # A. Static Score (Isolation Forest)
    # Pass .values to match the RobustScaler's fitted format (no feature names)
    X_static = scaler_static.transform(latest[static_cols].values.reshape(1, -1))
    raw_if_score = -iforest.decision_function(X_static)[0]
    iforest_norm = (raw_if_score - score_stats['if_mean']) / score_stats['if_std']
    
    # B. Temporal Score (LSTM)
    lstm_norm = 0.0
    has_temporal_context = len(history_for_sequence) >= 7
    if has_temporal_context:
        # Get the last 7 rows ending at the uploaded claim.
        recent_seq_df = history_for_sequence[seq_cols].tail(7)
        seq_scaled = scaler_seq.transform(recent_seq_df)
        X_seq = np.array([seq_scaled]) # Shape (1, 7, features)
        
        recon = lstm.predict(X_seq, verbose=0)
        raw_lstm_score = np.mean(np.square(X_seq - recon), axis=(1,2))[0]
        lstm_norm = (raw_lstm_score - score_stats['lstm_mean']) / score_stats['lstm_std']

    raw_hybrid = float(
        compute_hybrid_score(
            [iforest_norm],
            [lstm_norm],
            [has_temporal_context],
        )[0]
    )
    
    # THE MATHEMATICAL DECISION (Is it fraud?)
    is_anomaly = int(raw_hybrid > GLOBAL_THRESHOLD)
    alert_zone = classify_alert_zone(raw_hybrid)
    
    # Update the row for SHAP context
    latest['hybrid_score'] = raw_hybrid
    
    # ---------------------------------------------------------
    # 5. API RESPONSE & EXPLAINABILITY (XAI)
    # ---------------------------------------------------------
    result = {
        "transaction_id": int(latest.get('transaction_id', 0)),
        "provider_id": str(provider_id),
        "anomaly_score": float(round(raw_hybrid, 4)),
        "alert_zone": alert_zone,
        "is_normal": "No" if is_anomaly == 1 else "Yes"
    }
    
    if include_explainability and is_anomaly == 1:
        # Pass model and scaler to ensure SHAP works without re-loading files
        explanation_df = explain_transaction(latest, static_cols, iforest, scaler_static)
        explanation_text = explain_in_words(explanation_df)
        if alert_zone == "High Risk":
            prefix = "High-risk anomaly detected."
        else:
            prefix = "Suspicious anomaly detected."
        result["reason"] = f"{prefix} {str(explanation_text)}"
    elif alert_zone == "Suspicious":
        result["reason"] = "Pattern is near the anomaly boundary. Manual review is recommended."
    elif alert_zone == "High Risk":
        result["reason"] = "High-risk anomaly detected. The score strongly exceeds the learned anomaly boundary."
    else:
        result["reason"] = "Claim pattern is within the expected range."
        
    if not has_temporal_context:
        result["reason"] += " (Note: Limited provider history; LSTM analysis inactive)."

    if include_explainability:
        # EXPORT SHAP DATA FOR STREAMLIT PLOTLY CHART
        try:
            result["shap_data"] = get_shap_data_for_plotly(latest, static_cols, iforest, scaler_static)
            result["top_drivers"] = get_driver_bullets_from_shap_data(result["shap_data"])
        except Exception as e:
            print(f"Warning: SHAP Visualization failed: {e}")
            result["shap_data"] = None
            result["top_drivers"] = []
    else:
        result["shap_data"] = None
        result["top_drivers"] = []

    return result
