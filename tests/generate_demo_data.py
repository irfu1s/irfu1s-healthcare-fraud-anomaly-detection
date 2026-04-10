import os
import random
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "samples", "normal_demo_data.csv")

from src.data_loader import load_full_dataset
from src.model_inference import HIGH_RISK_ZONE_MIN, NORMAL_ZONE_MAX, run_inference

MAX_GENERATION_ATTEMPTS = 500

REQUIRED_COLUMNS = [
    "transaction_id",
    "patient_id",
    "provider_id",
    "claim_type",
    "service_start",
    "service_end",
    "admission_date",
    "billing_amount",
    "deductible_context",
    "coinsurance_context",
    "treatment_group",
    "diagnosis_context",
    "primary_procedure",
    "primary_service",
    "gender_context",
    "demographic_context",
    "has_diabetes",
    "has_chf",
    "has_cancer",
    "has_copd",
    "age",
]


def load_project_transaction_dataset():
    df = load_full_dataset()

    if "age" not in df.columns:
        df["age"] = pd.NA
    if "patient_age" in df.columns:
        df["age"] = df["age"].fillna(pd.to_numeric(df["patient_age"], errors="coerce"))
    if "patient_dob" in df.columns:
        patient_dob = pd.to_datetime(df["patient_dob"], errors="coerce")
        service_start = pd.to_datetime(df["service_start"], errors="coerce")
        derived_age = ((service_start - patient_dob).dt.days // 365.25).clip(lower=0)
        df["age"] = df["age"].fillna(derived_age)

    df["service_duration"] = (
        pd.to_datetime(df["service_end"], errors="coerce")
        - pd.to_datetime(df["service_start"], errors="coerce")
    ).dt.days.clip(lower=0)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Healthcare_transactions is missing columns: {missing}")

    for col in ["service_start", "service_end", "admission_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = [
        "billing_amount",
        "age",
        "provider_avg_billing",
        "provider_std_billing",
        "amount_vs_provider_avg",
        "rolling_std_7",
        "service_duration",
        "deductible_context",
        "coinsurance_context",
        "gender_context",
        "demographic_context",
        "has_diabetes",
        "has_chf",
        "has_cancer",
        "has_copd",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["provider_id"] = df["provider_id"].astype(str).str.strip().str.upper()
    df["patient_id"] = df["patient_id"].astype(str).str.strip().str.upper()
    df["claim_type"] = df["claim_type"].astype(str).str.strip()
    df["diagnosis_context"] = df["diagnosis_context"].fillna("UNKNOWN").astype(str).str.strip()
    df["primary_procedure"] = df["primary_procedure"].fillna("0").astype(str).str.strip()
    df["treatment_group"] = df["treatment_group"].fillna("0").astype(str).str.strip()
    df["primary_service"] = df["primary_service"].fillna("0").astype(str).str.strip()
    df = df.dropna(subset=["provider_id", "service_start", "billing_amount"])
    return df


def build_normal_pool(df):
    provider_history = df.groupby("provider_id")["transaction_id"].transform("count")
    eligible = df[provider_history >= 5].copy()

    billing_quantiles = (
        eligible.groupby("provider_id")["billing_amount"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q25_billing", 0.75: "q75_billing"})
    )
    eligible = eligible.merge(billing_quantiles, on="provider_id", how="left")

    provider_median = eligible.groupby("provider_id")["billing_amount"].transform("median")
    provider_std = eligible.groupby("provider_id")["billing_amount"].transform("std").fillna(0.0)

    allowed_gap = np.maximum.reduce(
        [
            np.full(len(eligible), 120.0),
            provider_median.abs().fillna(0.0).values * 0.08,
            provider_std.abs().fillna(0.0).values * 0.30,
        ]
    )

    deviation_mask = (eligible["billing_amount"] - provider_median).abs().values <= allowed_gap
    iqr_mask = (
        (eligible["billing_amount"] >= eligible["q25_billing"].fillna(eligible["billing_amount"]))
        & (eligible["billing_amount"] <= eligible["q75_billing"].fillna(eligible["billing_amount"]))
    )
    duration_mask = np.where(
        eligible["claim_type"].str.contains("Inpatient", case=False, na=False),
        eligible["service_duration"].fillna(0).between(1, 10),
        eligible["service_duration"].fillna(0).between(0, 1),
    )
    volatility_mask = True
    if "rolling_std_7" in eligible.columns:
        volatility_limit = np.maximum(
            provider_std.abs().fillna(0.0) * 1.10,
            50.0,
        )
        volatility_mask = eligible["rolling_std_7"].fillna(0.0) <= volatility_limit

    normal_pool = eligible[deviation_mask & iqr_mask & duration_mask & volatility_mask].copy()

    if normal_pool.empty or normal_pool["provider_id"].nunique() < 10:
        eligible["baseline_gap_abs"] = (eligible["billing_amount"] - provider_median).abs()
        fallback = (
            eligible.sort_values(["provider_id", "baseline_gap_abs"])
            .groupby("provider_id", group_keys=False)
            .head(3)
            .copy()
        )
        normal_pool = fallback

    return normal_pool


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
        "Label": "NORMAL",
        "scenario_profile": "normal_clear",
        "expected_zone": "Normal",
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


def _get_normal_band(profile, current_threshold):
    if profile == "clear":
        return 0.55, min(NORMAL_ZONE_MAX - 0.05, current_threshold - 0.45)
    if profile == "borderline":
        return NORMAL_ZONE_MAX + 0.02, min(current_threshold - 0.05, HIGH_RISK_ZONE_MIN - 0.30)
    raise ValueError(f"Unsupported normal profile: {profile}")


def generate_normal_data(row_count=20, seed=42, output_path=OUTPUT_PATH, profile="mixed"):
    if profile == "mixed":
        clear_rows = row_count // 2
        borderline_rows = row_count - clear_rows
        clear_df = generate_normal_data(
            row_count=clear_rows,
            seed=seed,
            output_path=None,
            profile="clear",
        )
        borderline_df = generate_normal_data(
            row_count=borderline_rows,
            seed=seed + 101,
            output_path=None,
            profile="borderline",
        )
        result = (
            pd.concat([clear_df, borderline_df], ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        result = _clean_output_dataframe(result)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.to_csv(output_path, index=False)
            print(f"Generated {len(result)} mixed normal rows at {output_path}")
        return result

    rng = random.Random(seed)
    df = load_project_transaction_dataset()
    normal_pool = build_normal_pool(df)
    current_threshold = float(pd.to_numeric(pd.Series([1.625]), errors="coerce").iloc[0])
    try:
        import joblib

        current_threshold = float(joblib.load(os.path.join(BASE_DIR, "models", "threshold.pkl")))
    except Exception:
        pass

    target_min_score, target_max_score = _get_normal_band(profile, current_threshold)
    target_max_score = max(target_min_score + 0.05, target_max_score)

    provider_stats = (
        normal_pool.groupby("provider_id")
        .agg(
            last_service_start=("service_start", "max"),
            median_billing=("billing_amount", "median"),
            q25_billing=("billing_amount", lambda x: x.quantile(0.25)),
            q75_billing=("billing_amount", lambda x: x.quantile(0.75)),
            median_deductible=("deductible_context", "median"),
            median_coinsurance=("coinsurance_context", "median"),
            visit_count=("transaction_id", "count"),
        )
        .reset_index()
    )
    provider_stats = provider_stats[provider_stats["visit_count"] >= 5]

    if provider_stats.empty:
        raise RuntimeError("Not enough provider history found in Healthcare_transactions.")

    eligible = normal_pool[normal_pool["provider_id"].isin(provider_stats["provider_id"])].copy()
    stats_lookup = provider_stats.set_index("provider_id").to_dict("index")

    rows = []
    best_fallback = []
    attempts = 0
    used_transaction_ids = set()
    used_patient_ids = set()

    while len(rows) < row_count and attempts < MAX_GENERATION_ATTEMPTS:
        attempts += 1
        source_row = eligible.sample(n=1, replace=True, random_state=seed + attempts).iloc[0]
        provider_id = source_row["provider_id"]
        provider_info = stats_lookup[provider_id]
        service_start = provider_info["last_service_start"] + timedelta(
            days=rng.randint(7, 28) if profile == "clear" else rng.randint(3, 18),
            hours=rng.randint(8, 16),
        )

        claim_type = str(source_row["claim_type"]).strip()
        if "Inpatient" in claim_type:
            stay_days = max(1, _safe_int((source_row["service_end"] - source_row["service_start"]).days, 3))
            service_end = service_start + timedelta(days=stay_days)
            deductible_context = max(_safe_float(source_row["deductible_context"]), provider_info["median_deductible"], 100.0)
            coinsurance_context = max(_safe_float(source_row["coinsurance_context"]), provider_info["median_coinsurance"], 40.0)
        else:
            claim_type = "Outpatient"
            service_end = service_start
            deductible_context = max(_safe_float(source_row["deductible_context"]), 0.0)
            coinsurance_context = max(_safe_float(source_row["coinsurance_context"]), 0.0)

        provider_median = max(_safe_float(provider_info["median_billing"], 50.0), 50.0)
        q25_billing = max(_safe_float(provider_info.get("q25_billing"), provider_median), 50.0)
        q75_billing = max(_safe_float(provider_info.get("q75_billing"), provider_median), q25_billing)
        provider_target = min(max(provider_median, q25_billing), q75_billing)
        billing_multiplier = rng.uniform(0.94, 0.99) if profile == "clear" else rng.uniform(0.995, 1.018)
        billing_amount = round(provider_target * billing_multiplier, 2)

        row = {
            "Label": "NORMAL",
            "scenario_profile": f"normal_{profile}",
            "expected_zone": "Normal" if profile == "clear" else "Suspicious",
            "transaction_id": _random_transaction_id(rng, service_start, used_transaction_ids),
            "patient_id": _unique_patient_id(rng, used_patient_ids),
            "provider_id": provider_id,
            "claim_type": claim_type,
            "service_start": service_start.strftime("%Y-%m-%d %H:%M:%S"),
            "service_end": service_end.strftime("%Y-%m-%d %H:%M:%S"),
            "admission_date": service_start.strftime("%Y-%m-%d"),
            "billing_amount": billing_amount,
            "deductible_context": round(deductible_context, 2),
            "coinsurance_context": round(coinsurance_context, 2),
            "treatment_group": _safe_text(source_row["treatment_group"], "0"),
            "diagnosis_context": _safe_text(source_row["diagnosis_context"], "UNKNOWN"),
            "primary_procedure": _safe_text(source_row["primary_procedure"], "0"),
            "primary_service": _safe_text(source_row["primary_service"], "0"),
            "patient_age": max(1, _safe_int(source_row["age"], rng.randint(25, 85))),
            "gender_context": _safe_int(source_row["gender_context"], rng.choice([0, 1])),
            "demographic_context": max(1, _safe_int(source_row["demographic_context"], 1)),
            "has_diabetes": min(1, max(0, _safe_int(source_row["has_diabetes"], 0))),
            "has_chf": min(1, max(0, _safe_int(source_row["has_chf"], 0))),
            "has_cancer": min(1, max(0, _safe_int(source_row["has_cancer"], 0))),
            "has_copd": min(1, max(0, _safe_int(source_row["has_copd"], 0))),
        }

        if row["claim_type"] == "Outpatient":
            row["service_end"] = row["service_start"]
            row["primary_procedure"] = row["primary_procedure"] or "0"
            row["treatment_group"] = row["treatment_group"] or "0"

        result = run_inference(row.copy())
        score = float(result.get("anomaly_score", 0.0))
        predicted_normal = str(result.get("is_normal", "Yes")).strip().lower() == "yes"
        gap_to_band = 0.0
        if score < target_min_score:
            gap_to_band = target_min_score - score
        elif score > target_max_score:
            gap_to_band = score - target_max_score

        best_fallback.append((gap_to_band, score, row.copy()))
        best_fallback = sorted(best_fallback, key=lambda item: (item[0], abs(item[1] - target_max_score)))[: row_count * 4]

        if predicted_normal and target_min_score <= score <= target_max_score:
            row["transaction_id"] = _random_transaction_id(rng, row["service_start"], used_transaction_ids)
            row["patient_id"] = _unique_patient_id(rng, used_patient_ids)
            rows.append(row)

    if len(rows) < row_count:
        for _, score, row in sorted(best_fallback, key=lambda item: (item[0], abs(item[1] - target_max_score))):
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
            f"Generated {len(result)} {profile} normal rows from Healthcare_transactions at {output_path} "
            f"(target anomaly_score band: {target_min_score:.2f} to {target_max_score:.2f})"
        )
    else:
        print(
            f"Generated {len(result)} {profile} normal rows from Healthcare_transactions "
            f"(target anomaly_score band: {target_min_score:.2f} to {target_max_score:.2f})"
        )
    return result


if __name__ == "__main__":
    generate_normal_data()
