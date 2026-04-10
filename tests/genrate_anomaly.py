import os
import random
import sys
from datetime import timedelta

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

FLAGGED_PATH = os.path.join(BASE_DIR, "data", "samples", "flagged_fraud_transactions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "samples", "anomaly_demo_data.csv")

from src.data_loader import load_full_dataset
from src.model_inference import HIGH_RISK_ZONE_MIN, run_inference

MAX_GENERATION_ATTEMPTS = 500

CLINICAL_MISMATCHES = [
    ("V222", "3340", "999"),
    ("V700", "44140", "998"),
    ("4011", "8154", "997"),
]


def load_source_data():
    if not os.path.exists(FLAGGED_PATH):
        raise RuntimeError(
            f"Run src\\scoring.py first so this file exists: {FLAGGED_PATH}"
    )

    flagged = pd.read_csv(FLAGGED_PATH, low_memory=False)
    features = load_full_dataset()

    for df in [flagged, features]:
        for col in ["service_start", "service_end", "admission_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "billing_amount" in df.columns:
            df["billing_amount"] = pd.to_numeric(df["billing_amount"], errors="coerce")

        df["provider_id"] = df["provider_id"].astype(str).str.strip().str.upper()

    flagged = flagged.dropna(subset=["provider_id", "service_start", "billing_amount"])
    features = features.dropna(subset=["provider_id", "service_start", "billing_amount"])
    return flagged, features


def _safe_int(value, default=0):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return int(numeric)


def _safe_float(value, default=0.0):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return float(numeric)


def _safe_text(value, default="0", uppercase=False):
    if value is None or pd.isna(value):
        text = default
    else:
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null", "nat"}:
            text = default
    return text.upper() if uppercase else text


def _clean_output_dataframe(df):
    df = df.copy()
    text_defaults = {
        "Label": "ANOMALY",
        "scenario_profile": "anomaly_extreme",
        "expected_zone": "High Risk",
        "patient_id": "UNKNOWN",
        "provider_id": "UNKNOWN",
        "claim_type": "Outpatient",
        "treatment_group": "0",
        "diagnosis_context": "UNKNOWN",
        "primary_procedure": "0",
        "primary_service": "0",
    }
    numeric_defaults = {
        "billing_amount": 0.0,
        "deductible_context": 0.0,
        "coinsurance_context": 0.0,
        "patient_age": 45,
        "gender_context": 1,
        "demographic_context": 1,
        "has_diabetes": 0,
        "has_chf": 0,
        "has_cancer": 0,
        "has_copd": 0,
    }

    for col, default in text_defaults.items():
        if col in df.columns:
            df[col] = df[col].map(lambda value: _safe_text(value, default))
    for col, default in numeric_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    for col in ["has_diabetes", "has_chf", "has_cancer", "has_copd"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=1).astype(int)
    for col in ["patient_age", "gender_context", "demographic_context"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if {"claim_type", "service_start", "service_end"}.issubset(df.columns):
        outpatient_mask = ~df["claim_type"].astype(str).str.contains("inpatient", case=False, na=False)
        df.loc[outpatient_mask, "service_end"] = df.loc[outpatient_mask, "service_start"]
        df.loc[outpatient_mask, ["primary_procedure", "treatment_group"]] = df.loc[
            outpatient_mask,
            ["primary_procedure", "treatment_group"],
        ].fillna("0")
    return df


def _random_patient_id(rng):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choices(chars, k=16))


def _random_transaction_id(rng, service_start, used_ids):
    while True:
        candidate = int("".join(rng.choices("123456789", k=1) + rng.choices("0123456789", k=13)))
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate


def _unique_patient_id(rng, used_ids):
    while True:
        candidate = _random_patient_id(rng)
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate


def _get_anomaly_band(profile, current_threshold):
    if profile == "borderline":
        return current_threshold + 0.05, HIGH_RISK_ZONE_MIN - 0.03
    if profile == "extreme":
        return HIGH_RISK_ZONE_MIN, HIGH_RISK_ZONE_MIN + 0.75
    raise ValueError(f"Unsupported anomaly profile: {profile}")


def generate_anomaly_data(row_count=20, seed=99, output_path=OUTPUT_PATH, profile="mixed"):
    if profile == "mixed":
        borderline_rows = row_count // 2
        extreme_rows = row_count - borderline_rows
        borderline_df = generate_anomaly_data(
            row_count=borderline_rows,
            seed=seed,
            output_path=None,
            profile="borderline",
        )
        extreme_df = generate_anomaly_data(
            row_count=extreme_rows,
            seed=seed + 101,
            output_path=None,
            profile="extreme",
        )
        result = (
            pd.concat([borderline_df, extreme_df], ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        result = _clean_output_dataframe(result)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.to_csv(output_path, index=False)
            print(f"Generated {len(result)} mixed anomaly rows at {output_path}")
        return result

    rng = random.Random(seed)
    flagged, features = load_source_data()
    current_threshold = 1.625
    try:
        import joblib

        current_threshold = float(joblib.load(os.path.join(BASE_DIR, "models", "threshold.pkl")))
    except Exception:
        pass

    target_min_score, target_max_score = _get_anomaly_band(profile, current_threshold)

    if flagged.empty:
        raise RuntimeError("flagged_fraud_transactions.csv is empty or missing usable anomaly rows.")

    provider_last_seen = (
        features.groupby("provider_id")["service_start"]
        .max()
        .to_dict()
    )

    rows = []
    anomaly_types = ["amount_spike", "velocity_burst", "clinical_mismatch", "phantom_stay"]
    best_fallback = []
    attempts = 0
    used_transaction_ids = set()
    used_patient_ids = set()

    while len(rows) < row_count and attempts < MAX_GENERATION_ATTEMPTS:
        attempts += 1
        source_row = flagged.sample(n=1, replace=True, random_state=seed + attempts).iloc[0]
        provider_id = source_row["provider_id"]
        anchor_time = provider_last_seen.get(provider_id, source_row["service_start"])
        if pd.isna(anchor_time):
            anchor_time = source_row["service_start"]
        if pd.isna(anchor_time):
            anchor_time = pd.Timestamp("2010-01-01 09:00:00")

        anomaly_type = rng.choice(anomaly_types)
        service_start = anchor_time + timedelta(days=rng.randint(1, 7), hours=rng.randint(1, 6))
        claim_type = "Inpatient" if "Inpatient" in str(source_row.get("claim_type", "")) else "Outpatient"
        billing_amount = max(_safe_float(source_row["billing_amount"], 500.0), 100.0)

        row = {
            "Label": "ANOMALY",
            "scenario_profile": f"anomaly_{profile}",
            "expected_zone": "Suspicious" if profile == "borderline" else "High Risk",
            "transaction_id": _random_transaction_id(rng, service_start, used_transaction_ids),
            "patient_id": _unique_patient_id(rng, used_patient_ids),
            "provider_id": provider_id,
            "claim_type": claim_type,
            "service_start": service_start.strftime("%Y-%m-%d %H:%M:%S"),
            "service_end": service_start.strftime("%Y-%m-%d %H:%M:%S"),
            "admission_date": service_start.strftime("%Y-%m-%d"),
            "billing_amount": round(billing_amount, 2),
            "deductible_context": round(max(_safe_float(source_row.get("deductible_context"), 0.0), 0.0), 2),
            "coinsurance_context": round(max(_safe_float(source_row.get("coinsurance_context"), 0.0), 0.0), 2),
            "treatment_group": _safe_text(source_row.get("treatment_group"), "0"),
            "diagnosis_context": _safe_text(source_row.get("diagnosis_context"), "UNKNOWN"),
            "primary_procedure": _safe_text(source_row.get("primary_procedure"), "0"),
            "primary_service": _safe_text(source_row.get("primary_service"), "0"),
            "patient_age": max(1, _safe_int(source_row.get("age"), rng.randint(25, 85))),
            "gender_context": min(1, max(0, _safe_int(source_row.get("gender_context"), rng.choice([0, 1])))),
            "demographic_context": max(1, _safe_int(source_row.get("demographic_context"), 1)),
            "has_diabetes": min(1, max(0, _safe_int(source_row.get("has_diabetes"), 0))),
            "has_chf": min(1, max(0, _safe_int(source_row.get("has_chf"), 0))),
            "has_cancer": min(1, max(0, _safe_int(source_row.get("has_cancer"), 0))),
            "has_copd": min(1, max(0, _safe_int(source_row.get("has_copd"), 0))),
        }

        if anomaly_type == "amount_spike":
            multiplier = rng.uniform(1.08, 1.22) if profile == "borderline" else rng.uniform(1.35, 1.90)
            row["billing_amount"] = round(billing_amount * multiplier, 2)
            if row["claim_type"] == "Inpatient":
                row["deductible_context"] = max(row["deductible_context"], 1024.0)
                row["coinsurance_context"] = max(row["coinsurance_context"], 64.0 if profile == "borderline" else 128.0)
                stay_days = rng.randint(2, 4) if profile == "borderline" else rng.randint(5, 10)
                row["service_end"] = (service_start + timedelta(days=stay_days)).strftime("%Y-%m-%d %H:%M:%S")
        elif anomaly_type == "velocity_burst":
            burst_time = anchor_time + timedelta(hours=rng.randint(1, 3))
            row["service_start"] = burst_time.strftime("%Y-%m-%d %H:%M:%S")
            row["service_end"] = (burst_time + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S")
            row["billing_amount"] = round(
                billing_amount * (rng.uniform(1.04, 1.14) if profile == "borderline" else rng.uniform(1.15, 1.30)),
                2,
            )
        elif anomaly_type == "clinical_mismatch":
            mismatch_dx, mismatch_px, mismatch_tg = rng.choice(CLINICAL_MISMATCHES)
            row["claim_type"] = "Outpatient"
            row["billing_amount"] = round(
                billing_amount * (rng.uniform(1.10, 1.30) if profile == "borderline" else rng.uniform(1.35, 1.70)),
                2,
            )
            row["diagnosis_context"] = mismatch_dx
            row["primary_procedure"] = mismatch_px
            row["treatment_group"] = mismatch_tg
            row["deductible_context"] = 0.0
            row["coinsurance_context"] = max(row["coinsurance_context"], 30.0 if profile == "borderline" else 60.0)
            row["service_end"] = row["service_start"]
        else:
            row["claim_type"] = "Inpatient"
            row["billing_amount"] = round(
                billing_amount * (rng.uniform(1.15, 1.35) if profile == "borderline" else rng.uniform(1.45, 2.00)),
                2,
            )
            row["diagnosis_context"] = row["diagnosis_context"] or "4280"
            row["primary_procedure"] = row["primary_procedure"] or "3722"
            row["treatment_group"] = row["treatment_group"] or "089"
            row["deductible_context"] = max(row["deductible_context"], 1024.0)
            row["coinsurance_context"] = max(row["coinsurance_context"], 64.0 if profile == "borderline" else 128.0)
            row["service_end"] = row["service_start"]

        if row["claim_type"] == "Outpatient":
            row["service_end"] = row["service_start"]
            row["primary_procedure"] = row["primary_procedure"] or "0"
            row["treatment_group"] = row["treatment_group"] or "0"

        result = run_inference(row.copy())
        score = float(result.get("anomaly_score", 0.0))
        predicted_anomaly = str(result.get("is_normal", "Yes")).strip().lower() == "no"
        gap_to_band = 0.0
        if score < target_min_score:
            gap_to_band = target_min_score - score
        elif score > target_max_score:
            gap_to_band = score - target_max_score

        best_fallback.append((gap_to_band, score, row.copy()))
        best_fallback = sorted(best_fallback, key=lambda item: (item[0], abs(item[1] - target_min_score)))[: row_count * 4]

        if predicted_anomaly and target_min_score <= score <= target_max_score:
            row["transaction_id"] = _random_transaction_id(rng, row["service_start"], used_transaction_ids)
            row["patient_id"] = _unique_patient_id(rng, used_patient_ids)
            rows.append(row)

    if len(rows) < row_count:
        for _, score, row in sorted(best_fallback, key=lambda item: (item[0], abs(item[1] - target_min_score))):
            if len(rows) >= row_count:
                break
            row["transaction_id"] = _random_transaction_id(rng, row["service_start"], used_transaction_ids)
            row["patient_id"] = _unique_patient_id(rng, used_patient_ids)
            rows.append(row)

    result = _clean_output_dataframe(pd.DataFrame(rows))
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(
            f"Generated {len(result)} {profile} anomaly rows from flagged_fraud_transactions.csv at {output_path} "
            f"using Healthcare_transactions provider history "
            f"(target anomaly_score band: {target_min_score:.2f} to {target_max_score:.2f})"
        )
    else:
        print(
            f"Generated {len(result)} {profile} anomaly rows from flagged_fraud_transactions.csv "
            f"using Healthcare_transactions provider history "
            f"(target anomaly_score band: {target_min_score:.2f} to {target_max_score:.2f})"
        )
    return result


if __name__ == "__main__":
    generate_anomaly_data()
