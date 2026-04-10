import pandas as pd
import numpy as np

# Silence the annoying Pandas Future Warning globally
pd.set_option('future.no_silent_downcasting', True)

STATIC_FEATURE_COLUMNS = [
    'billing_amount_log',
    'deductible_context',
    'coinsurance_context',
    'provider_avg_billing',
    'provider_std_billing',
    'provider_deviation_score',
    'service_duration',
    'age',
    'rolling_mean_7',
    'rolling_std_7',
    'has_diabetes',
    'has_chf',
    'has_cancer',
    'has_copd',
    'patient_hoarding_index',
    'dx_px_combo_spike'
]

SEQUENCE_FEATURE_COLUMNS = [
    'billing_amount_log',
    'time_gap',
    'services_per_day',
    'rolling_mean_7',
    'rolling_activity_7'
]

DEFAULT_COLUMN_VALUES = {
    'billing_amount': 0.0,
    'deductible_context': 0.0,
    'coinsurance_context': 0.0,
    'claim_type': 'UNKNOWN',
    'diagnosis_context': 'UNKNOWN',
    'primary_procedure': 'NONE',
    'provider_id': 'UNKNOWN',
    'patient_id': 'UNKNOWN',
    'transaction_id': 0,
    'has_diabetes': 0,
    'has_chf': 0,
    'has_cancer': 0,
    'has_copd': 0
}


def _ensure_required_columns(df):
    df = df.copy()

    for col, default in DEFAULT_COLUMN_VALUES.items():
        if col not in df.columns:
            df[col] = default

    if 'service_start' not in df.columns:
        df['service_start'] = pd.NaT
    if 'service_end' not in df.columns:
        df['service_end'] = df['service_start']

    if 'age' not in df.columns:
        df['age'] = np.nan

    # Support historical tables that only store engineered log amounts.
    if 'billing_amount_log' in df.columns:
        missing_billing = df['billing_amount'].isna() | (df['billing_amount'] == 0)
        recovered_billing = np.expm1(pd.to_numeric(df['billing_amount_log'], errors='coerce'))
        df.loc[missing_billing, 'billing_amount'] = recovered_billing.loc[missing_billing]

    numeric_cols = [
        'billing_amount', 'deductible_context', 'coinsurance_context', 'transaction_id',
        'has_diabetes', 'has_chf', 'has_cancer', 'has_copd', 'age'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['provider_id'] = df['provider_id'].astype(str).str.strip()
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    df['claim_type'] = (
        df['claim_type']
        .astype(str)
        .str.strip()
        .replace(
            {
                'Outpatient (Same Day)': 'Outpatient',
                'Inpatient (Admitted)': 'Inpatient'
            }
        )
    )
    df['diagnosis_context'] = df['diagnosis_context'].astype(str).str.strip().replace({'': 'UNKNOWN'})
    df['primary_procedure'] = df['primary_procedure'].astype(str).str.strip().replace({'': 'NONE'})

    return df


def build_features(df):
    df = _ensure_required_columns(df)

    # ---------- 1. TIME ----------
    df['service_start'] = pd.to_datetime(df['service_start'], errors='coerce')
    df['service_end'] = pd.to_datetime(df['service_end'], errors='coerce')
    df['service_end'] = df['service_end'].fillna(df['service_start'])
    df['service_duration'] = (df['service_end'] - df['service_start']).dt.days.clip(lower=0)

    df['day_of_week'] = df['service_start'].dt.dayofweek
    df['month'] = df['service_start'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    # ---------- 2. AGE → AGE BAND ----------
    if 'patient_dob' in df.columns:
        df['patient_dob'] = pd.to_datetime(df['patient_dob'], errors='coerce')
        hist_age = ((df['service_start'] - df['patient_dob']).dt.days // 365.25).clip(lower=0)
        df['age'] = hist_age if 'age' not in df.columns else df['age'].fillna(hist_age)
            
    if 'patient_age' in df.columns:
        df['age'] = df['patient_age'] if 'age' not in df.columns else df['age'].fillna(df['patient_age'])

    conditions = [df['age'].isna(), df['age'] <= 18, df['age'] <= 40, df['age'] <= 60]
    choices = ["unknown", "child", "young", "middle"]
    df['age_band'] = np.select(conditions, choices, default="senior")

    # ---------- 3. CHRONOLOGICAL SORTING ----------
    df = df.sort_values(['provider_id', 'service_start']).reset_index(drop=True)

    # ---------- 4. THE MAGIC FIX: CLIPPING & LOG TRANSFORM ----------
    # 1. Clip the extreme top 1% of raw billing to prevent mathematical distortion
    p99 = np.percentile(df['billing_amount'].fillna(0), 99)
    df['billing_amount'] = np.clip(df['billing_amount'], 0, p99)

    # 2. Transform into Log-Space. ALL subsequent math MUST use this column!
    df['billing_amount_log'] = np.log1p(df['billing_amount'])

    # ---------- 5. SECURE BASELINE MATH (NOW IN LOG-SPACE) ----------
    global_avg_log = df['billing_amount_log'].mean()
    global_std_log = df['billing_amount_log'].std()

    shifted_billing_log = df.groupby('provider_id')['billing_amount_log'].shift(1)

    df['provider_avg_billing'] = shifted_billing_log.groupby(df['provider_id']).expanding().mean().reset_index(level=0, drop=True).fillna(global_avg_log)
    df['provider_std_billing'] = shifted_billing_log.groupby(df['provider_id']).expanding().std().reset_index(level=0, drop=True).fillna(global_std_log)

    # --- THE 3-LINE KILL-SHOT FEATURE ---
    df['provider_deviation_score'] = (df['billing_amount_log'] - df['provider_avg_billing']) / (df['provider_std_billing'] + 1e-6)
    df['provider_deviation_score'] = np.clip(df['provider_deviation_score'], -10, 10)

    df['provider_age_avg'] = df.groupby(['provider_id', 'age_band'])['billing_amount_log'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['provider_avg_billing'])

    # ---------- 5.5. CLINICAL TEXT ENCODING (IN LOG-SPACE) ----------
    df['claim_type_avg'] = df.groupby(['provider_id', 'claim_type'])['billing_amount_log'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['provider_avg_billing'])

    df['diagnosis_avg'] = df.groupby(['provider_id', 'diagnosis_context'])['billing_amount_log'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['claim_type_avg'])

    df['primary_procedure'] = df['primary_procedure'].fillna("NONE")
    df['procedure_avg'] = df.groupby(['provider_id', 'primary_procedure'])['billing_amount_log'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['claim_type_avg'])

    df['dx_px_combo'] = df['diagnosis_context'].astype(str) + "_" + df['primary_procedure'].astype(str)
    
    df['dx_px_combo_avg'] = df.groupby(['provider_id', 'dx_px_combo'])['billing_amount_log'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['diagnosis_avg'])

    # ---------- 6. CONTEXTUAL DEVIATION (COMPARING LOGS) ----------
    df['amount_vs_provider'] = df['billing_amount_log'] - df['provider_avg_billing']
    df['amount_vs_provider_age'] = df['billing_amount_log'] - df['provider_age_avg']
    df['amount_vs_claim_type'] = df['billing_amount_log'] - df['claim_type_avg']
    df['amount_vs_diagnosis'] = df['billing_amount_log'] - df['diagnosis_avg']
    df['amount_vs_procedure'] = df['billing_amount_log'] - df['procedure_avg']
    
    # EXACT MATCH TO TRAINING ARRAY REQUIRED FOR CLINICAL MISMATCH DETECTOR
    df['dx_px_combo_spike'] = df['billing_amount_log'] - df['dx_px_combo_avg']

    df['relative_provider_deviation'] = np.where(
        df['provider_avg_billing'] != 0,
        df['amount_vs_provider'] / df['provider_avg_billing'],
        0
    )

    # ---------- 7. PATIENT HOARDING & WEEKEND SPIKES ----------
    df['is_new_patient'] = (~df.duplicated(subset=['provider_id', 'patient_id'])).astype(int)
    df['unique_patients_to_date'] = df.groupby('provider_id')['is_new_patient'].cumsum()
    df['total_visits_to_date'] = df.groupby('provider_id').cumcount() + 1
    
    df['patient_hoarding_index'] = np.where(
        df['total_visits_to_date'] > 5, 
        df['unique_patients_to_date'] / df['total_visits_to_date'], 
        1.0 
    )

    df['weekend_avg_billing'] = df.groupby(['provider_id', 'is_weekend'])['billing_amount_log'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(global_avg_log)
    
    df['weekend_fraud_spike'] = np.where(
        df['is_weekend'] == 1,
        df['billing_amount_log'] - df['weekend_avg_billing'],
        0
    )

    # ---------- 8. TEMPORAL BEHAVIOR (THE LSTM FIX) ----------
    # Calculate time gap in HOURS
    df['time_gap'] = df.groupby('provider_id')['service_start'].diff().dt.total_seconds() / 3600.0
    # Cap at 720 hours (30 days)
    df['time_gap'] = df['time_gap'].fillna(0).clip(lower=0, upper=720) 

    df['service_date'] = df['service_start'].dt.date
    df['services_per_day'] = df.groupby(['provider_id', 'service_date'])['transaction_id'].transform('count')

    # ---------- 9. SECURE ROLLING AVERAGES ----------
    df['rolling_mean_7'] = shifted_billing_log.groupby(df['provider_id']).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True).fillna(df['provider_avg_billing'])
    df['rolling_std_7'] = shifted_billing_log.groupby(df['provider_id']).rolling(7, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    
    shifted_activity = df.groupby('provider_id')['services_per_day'].shift(1)
    df['rolling_activity_7'] = shifted_activity.groupby(df['provider_id']).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True).fillna(1)

    # ---------- 10. CLEAN UP ----------
    cols_to_drop = [
        'is_new_patient', 'unique_patients_to_date', 'total_visits_to_date', 
        'weekend_avg_billing', 'dx_px_combo', 'dx_px_combo_avg',
        'claim_type_avg', 'diagnosis_avg', 'procedure_avg', 'service_date'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    df = df.replace([np.inf, -np.inf], 0.0).infer_objects(copy=False)
    df = df.fillna(0) 

    return df
