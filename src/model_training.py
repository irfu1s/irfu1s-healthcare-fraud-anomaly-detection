import os
# Suppress the annoying TensorFlow warning immediately
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from src.data_loader import load_full_dataset
from src.feature_engineering import (
    STATIC_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_COLUMNS,
    build_features,
)
from src.hybrid_scoring import IFOREST_CONTAMINATION, LSTM_NORMAL_FILTER_PERCENTILE

# ---------- 1. LOAD + FEATURE ----------
print("Loading data...")
df = load_full_dataset()
print("Building features...")
df = build_features(df)

# CRITICAL FIX 1: Sort the dataframe BEFORE extracting arrays!
if 'service_start' in df.columns:
    df = df.sort_values(['provider_id', 'service_start']).reset_index(drop=True)
else:
    df = df.sort_values(['provider_id']).reset_index(drop=True)

# ---------- 2. STATIC FEATURES (Isolation Forest) ----------
static_cols = STATIC_FEATURE_COLUMNS

X_static = df[static_cols].values

print("Scaling static features...")
scaler_static = RobustScaler()
X_static_scaled = scaler_static.fit_transform(X_static)

# ---------- 3. TRAIN ISOLATION FOREST ----------
print("Training Isolation Forest...")
# Tuned higher to improve anomaly recall without collapsing into over-flagging.
iso_forest = IsolationForest(
    contamination=IFOREST_CONTAMINATION,
    n_estimators=200,
    random_state=42,
)
iso_forest.fit(X_static_scaled)

os.makedirs('models', exist_ok=True)
joblib.dump(iso_forest, 'models/iforest_model.pkl')
joblib.dump(scaler_static, 'models/scaler_static.pkl')

# ---------- 4. TEMPORAL FEATURES (LSTM) ----------
seq_cols = SEQUENCE_FEATURE_COLUMNS

print("Scaling temporal sequences...")
scaler_seq = RobustScaler()
df[seq_cols] = scaler_seq.fit_transform(df[seq_cols])
joblib.dump(scaler_seq, 'models/scaler_seq.pkl') 

# Filter out the worst static anomalies so the LSTM only learns "normal" behavior
if_scores = iso_forest.decision_function(X_static_scaled)
normal_mask = if_scores > np.percentile(if_scores, LSTM_NORMAL_FILTER_PERCENTILE)
df_normal = df[normal_mask].copy()

sequences = []
window = 7

print("Building sequences from strictly NORMAL data... this might take a minute.")
for provider_id, group in df_normal.groupby('provider_id'):
    data = group[seq_cols].values
    
    if len(data) < window:
        continue
        
    for i in range(window, len(data) + 1):
        sequences.append(data[i-window : i])

X_seq = np.array(sequences)

if len(X_seq) == 0:
    print("WARNING: Not enough sequences found. Falling back to simple default shape.")
    X_seq = np.zeros((10, window, len(seq_cols)))

# ---------- 5. BUILD LSTM AUTOENCODER ----------
timesteps = X_seq.shape[1]
features = X_seq.shape[2]

inputs = Input(shape=(timesteps, features))
encoded = LSTM(48, return_sequences=False)(inputs)       
repeated = RepeatVector(timesteps)(encoded)              
decoded = LSTM(48, return_sequences=True)(repeated)      
outputs = TimeDistributed(Dense(features))(decoded)      

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# ---------- 6. TRAIN ----------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Starting Autoencoder training...")
autoencoder.fit(
    X_seq, X_seq,
    epochs=15, batch_size=32, validation_split=0.1, shuffle=True, callbacks=[early_stop] 
)

autoencoder.save('models/lstm_autoencoder.keras') 
print("\n✅ Training Complete. All models and scalers safely exported.")
