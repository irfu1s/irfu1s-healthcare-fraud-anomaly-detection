import os
# Suppress the annoying TensorFlow warning immediately
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from src.feature_engineering import STATIC_FEATURE_COLUMNS, SEQUENCE_FEATURE_COLUMNS
from src.hybrid_scoring import HYBRID_THRESHOLD_PERCENTILE, compute_hybrid_score

# ---------- LOAD MODELS ----------
print("Loading trained models and scalers...")
iforest = joblib.load('models/iforest_model.pkl')
scaler_static = joblib.load('models/scaler_static.pkl')
lstm = load_model('models/lstm_autoencoder.keras')
scaler_seq = joblib.load('models/scaler_seq.pkl')

def get_iforest_score(df, static_cols):
    X = scaler_static.transform(df[static_cols].values)
    raw_score = iforest.decision_function(X)
    return -raw_score

def get_lstm_score(X_seq):
    recon = lstm.predict(X_seq, verbose=0)
    error = np.mean(np.square(X_seq - recon), axis=(1,2))
    return error

def score_dataset(df, static_cols, seq_cols, window=7):
    # Note: We removed the redundant np.clip() and sorting here because 
    # feature_engineering.py now perfectly handles clipping AND log transformations.

    df['iforest_score'] = get_iforest_score(df, static_cols)
    df['lstm_score'] = 0.0 
    df['has_temporal_context'] = 0
    
    df_scaled_seq = df.copy()
    df_scaled_seq[seq_cols] = scaler_seq.transform(df[seq_cols])
    
    print("Calculating LSTM sequence scores per provider...")
    for provider_id, group in df_scaled_seq.groupby('provider_id'):
        data = group[seq_cols].values
        
        if len(data) < window:
            continue
            
        sequences = np.array([data[i-window : i] for i in range(window, len(data) + 1)])
        provider_lstm_scores = get_lstm_score(sequences)
        
        for idx, score in zip(group.index[window-1:], provider_lstm_scores):
            df.at[idx, 'lstm_score'] = score
            df.at[idx, 'has_temporal_context'] = 1

    df['lstm_score'] = df['lstm_score'].fillna(0)

    if_clipped = np.clip(df['iforest_score'], a_min=None, a_max=np.percentile(df['iforest_score'], 99))
    lstm_clipped = np.clip(df['lstm_score'], a_min=None, a_max=np.percentile(df['lstm_score'], 99))
    
    score_stats = {
        'if_mean': if_clipped.mean(), 'if_std': if_clipped.std() + 1e-9,
        'lstm_mean': lstm_clipped.mean(), 'lstm_std': lstm_clipped.std() + 1e-9
    }
    joblib.dump(score_stats, 'models/score_stats.pkl')

    df['iforest_norm'] = (if_clipped - score_stats['if_mean']) / score_stats['if_std']
    df['lstm_norm'] = (lstm_clipped - score_stats['lstm_mean']) / score_stats['lstm_std']
    raw_hybrid = compute_hybrid_score(
        df['iforest_norm'].values,
        df['lstm_norm'].values,
        df['has_temporal_context'].values,
    )
    
    # KEEP HYBRID SCORE RAW for mathematically correct thresholding!
    df['hybrid_score'] = raw_hybrid
    
    return df

def apply_threshold(df, percentile=HYBRID_THRESHOLD_PERCENTILE):
    threshold = np.percentile(df['hybrid_score'], percentile)
    joblib.dump(threshold, 'models/threshold.pkl')
    df['is_anomaly'] = (df['hybrid_score'] > threshold).astype(int)
    return df, threshold

if __name__ == "__main__":
    from src.data_loader import load_full_dataset
    from src.feature_engineering import build_features

    # THE SYNCHRONIZED FIX: 16 clean features matching train.py perfectly
    static_cols = STATIC_FEATURE_COLUMNS

    seq_cols = SEQUENCE_FEATURE_COLUMNS

    print("Loading data from database...")
    df = load_full_dataset()
    
    print("Building features...")
    df = build_features(df)

    print("Scoring dataset... (This will take a moment)")
    df = score_dataset(df, static_cols, seq_cols)

    print("Applying threshold...")
    df, threshold = apply_threshold(df, percentile=HYBRID_THRESHOLD_PERCENTILE)

    print(f"\n--- SCORING COMPLETE ---")
    print(f"Calculated {HYBRID_THRESHOLD_PERCENTILE}th Percentile Threshold: {threshold:.4f}")
    print(f"Total Anomalies Flagged: {df['is_anomaly'].sum()}")
    print("Total rows:", len(df))
    print("Anomaly %:", df['is_anomaly'].mean() * 100)
    
    # Safely ensure the directory exists before saving
    os.makedirs("data/samples", exist_ok=True)
    anomalies = df[df['is_anomaly'] == 1]
    anomalies.to_csv("data/samples/flagged_fraud_transactions.csv", index=False)
