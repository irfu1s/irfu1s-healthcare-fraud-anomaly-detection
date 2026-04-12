import argparse
import os
import random
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "samples", "mixed_eval_data.csv")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_full_dataset


OUTPUT_COLUMNS = [
    "Label",
    "scenario_profile",
    "expected_zone",
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
    "patient_age",
    "gender_context",
    "demographic_context",
    "has_diabetes",
    "has_chf",
    "has_cancer",
    "has_copd",
]


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
    text_defaults = {
        "Label": "UNKNOWN",
        "scenario_profile": "unknown",
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

    df = df.copy()
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

    inpatient_mask = df["claim_type"].astype(str).str.contains("inpatient", case=False, na=False)
    outpatient_mask = ~inpatient_mask
    df.loc[outpatient_mask, "service_end"] = df.loc[outpatient_mask, "service_start"]
    df.loc[outpatient_mask, ["primary_procedure", "treatment_group"]] = df.loc[
        outpatient_mask,
        ["primary_procedure", "treatment_group"],
    ].fillna("0")
    df.loc[inpatient_mask, "primary_procedure"] = df.loc[inpatient_mask, "primary_procedure"].replace("", "1000")
    df.loc[inpatient_mask, "treatment_group"] = df.loc[inpatient_mask, "treatment_group"].replace("", "100")
    return df


def _random_patient_id(rng):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choices(chars, k=16))


def _unique_transaction_id(rng, used_ids):
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


def _load_raw_transactions():
    df = load_full_dataset()

    for col in ["service_start", "service_end", "admission_date", "patient_dob"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = [
        "billing_amount",
        "deductible_context",
        "coinsurance_context",
        "service_duration",
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

    if "patient_dob" in df.columns:
        age = ((df["service_start"] - df["patient_dob"]).dt.days / 365.25).round()
        df["patient_age"] = age.clip(lower=1, upper=105)
    elif "patient_age" not in df.columns:
        df["patient_age"] = pd.NA

    df["provider_id"] = df["provider_id"].astype(str).str.strip().str.upper()
    df["patient_id"] = df["patient_id"].astype(str).str.strip().str.upper()
    df["claim_type"] = df["claim_type"].astype(str).str.strip()
    df["diagnosis_context"] = df["diagnosis_context"].fillna("UNKNOWN").astype(str).str.strip()
    df["primary_procedure"] = df["primary_procedure"].fillna("0").astype(str).str.strip()
    df["treatment_group"] = df["treatment_group"].fillna("0").astype(str).str.strip()
    df["primary_service"] = df["primary_service"].fillna("0").astype(str).str.strip()

    df = df.dropna(subset=["provider_id", "service_start", "billing_amount"])
    df = df[df["billing_amount"] > 0].copy()
    return df


def _build_provider_context(df):
    context = (
        df.groupby("provider_id")
        .agg(
            provider_count=("transaction_id", "count"),
            last_service_start=("service_start", "max"),
            median_billing=("billing_amount", "median"),
            q25_billing=("billing_amount", lambda x: x.quantile(0.25)),
            q75_billing=("billing_amount", lambda x: x.quantile(0.75)),
            std_billing=("billing_amount", "std"),
            median_deductible=("deductible_context", "median"),
            median_coinsurance=("coinsurance_context", "median"),
            median_duration=("service_duration", "median"),
        )
        .reset_index()
    )
    return context[context["provider_count"] >= 10].copy()


def _build_normal_pool(df, provider_context):
    eligible = df[df["provider_id"].isin(provider_context["provider_id"])].copy()
    stats = provider_context.set_index("provider_id")
    eligible["provider_median_billing"] = eligible["provider_id"].map(stats["median_billing"])
    eligible["provider_q25_billing"] = eligible["provider_id"].map(stats["q25_billing"])
    eligible["provider_q75_billing"] = eligible["provider_id"].map(stats["q75_billing"])

    amount = eligible["billing_amount"]
    baseline = eligible["provider_median_billing"].fillna(amount)
    q25 = eligible["provider_q25_billing"].fillna(amount)
    q75 = eligible["provider_q75_billing"].fillna(amount)
    close_to_baseline = (amount - baseline).abs() <= np.maximum(150.0, baseline.abs() * 0.12)
    inside_iqr = amount.between(q25, q75)

    outpatient = eligible["claim_type"].str.contains("Outpatient", case=False, na=False)
    duration = eligible["service_duration"].fillna(0)
    normal_duration = np.where(outpatient, duration.between(0, 1), duration.between(1, 10))

    pool = eligible[close_to_baseline & inside_iqr & normal_duration].copy()
    if len(pool) < 100:
        pool = eligible.sort_values("billing_amount").groupby("provider_id", group_keys=False).head(3).copy()
    return pool


def _base_output_row(source_row, provider_info, label, scenario_profile, expected_zone, rng, used_txn_ids, used_patient_ids):
    last_seen = provider_info.get("last_service_start", source_row["service_start"])
    if pd.isna(last_seen):
        last_seen = source_row["service_start"]
    service_start = pd.to_datetime(last_seen) + timedelta(days=rng.randint(1, 21), hours=rng.randint(8, 16))

    claim_type = "Inpatient" if "inpatient" in str(source_row.get("claim_type", "")).lower() else "Outpatient"
    stay_days = max(0, _safe_int(source_row.get("service_duration"), 0))
    if claim_type == "Inpatient":
        stay_days = max(1, min(stay_days or rng.randint(2, 5), 10))
        service_end = service_start + timedelta(days=stay_days)
    else:
        service_end = service_start

    row = {
        "Label": label,
        "scenario_profile": scenario_profile,
        "expected_zone": expected_zone,
        "transaction_id": _unique_transaction_id(rng, used_txn_ids),
        "patient_id": _unique_patient_id(rng, used_patient_ids),
        "provider_id": _safe_text(source_row["provider_id"], "UNKNOWN", uppercase=True),
        "claim_type": claim_type,
        "service_start": service_start,
        "service_end": service_end,
        "admission_date": service_start.date(),
        "billing_amount": round(max(_safe_float(source_row.get("billing_amount"), 100.0), 20.0), 2),
        "deductible_context": round(max(_safe_float(source_row.get("deductible_context"), 0.0), 0.0), 2),
        "coinsurance_context": round(max(_safe_float(source_row.get("coinsurance_context"), 0.0), 0.0), 2),
        "treatment_group": _safe_text(source_row.get("treatment_group"), "0"),
        "diagnosis_context": _safe_text(source_row.get("diagnosis_context"), "UNKNOWN"),
        "primary_procedure": _safe_text(source_row.get("primary_procedure"), "0"),
        "primary_service": _safe_text(source_row.get("primary_service"), "0"),
        "patient_age": max(1, min(_safe_int(source_row.get("patient_age"), rng.randint(25, 85)), 105)),
        "gender_context": min(1, max(0, _safe_int(source_row.get("gender_context"), rng.choice([0, 1])))),
        "demographic_context": max(1, _safe_int(source_row.get("demographic_context"), 1)),
        "has_diabetes": min(1, max(0, _safe_int(source_row.get("has_diabetes"), 0))),
        "has_chf": min(1, max(0, _safe_int(source_row.get("has_chf"), 0))),
        "has_cancer": min(1, max(0, _safe_int(source_row.get("has_cancer"), 0))),
        "has_copd": min(1, max(0, _safe_int(source_row.get("has_copd"), 0))),
    }
    return row


def _finalize_dates(row):
    for col in ["service_start", "service_end"]:
        row[col] = pd.to_datetime(row[col]).strftime("%Y-%m-%d %H:%M:%S")
    row["admission_date"] = pd.to_datetime(row["admission_date"]).strftime("%Y-%m-%d")
    return row


def _make_normal_rows(pool, provider_context, count, profile, seed):
    rng = random.Random(seed)
    used_txn_ids = set()
    used_patient_ids = set()
    provider_lookup = provider_context.set_index("provider_id").to_dict("index")
    rows = []

    for i in range(count):
        source_row = pool.sample(n=1, replace=True, random_state=seed + i + 1).iloc[0]
        provider_info = provider_lookup[source_row["provider_id"]]
        row = _base_output_row(
            source_row,
            provider_info,
            "NORMAL",
            f"normal_{profile}",
            "Normal" if profile == "clear" else "Suspicious",
            rng,
            used_txn_ids,
            used_patient_ids,
        )

        median_billing = max(_safe_float(provider_info.get("median_billing"), row["billing_amount"]), 20.0)
        if profile == "clear":
            row["billing_amount"] = round(median_billing * rng.uniform(0.92, 1.03), 2)
            row["deductible_context"] = round(max(_safe_float(provider_info.get("median_deductible"), 0.0), 0.0), 2)
            row["coinsurance_context"] = round(max(_safe_float(provider_info.get("median_coinsurance"), 0.0), 0.0), 2)
        else:
            row["billing_amount"] = round(median_billing * rng.uniform(1.03, 1.16), 2)
            row["deductible_context"] = round(max(row["deductible_context"], _safe_float(provider_info.get("median_deductible"), 0.0)), 2)
            row["coinsurance_context"] = round(max(row["coinsurance_context"], _safe_float(provider_info.get("median_coinsurance"), 0.0)), 2)

        if row["claim_type"] == "Outpatient":
            row["service_end"] = row["service_start"]

        rows.append(_finalize_dates(row))

    return rows


def _make_anomaly_rows(source_df, provider_context, count, profile, seed):
    rng = random.Random(seed)
    used_txn_ids = set()
    used_patient_ids = set()
    provider_lookup = provider_context.set_index("provider_id").to_dict("index")
    eligible = source_df[source_df["provider_id"].isin(provider_context["provider_id"])].copy()
    rows = []

    for i in range(count):
        source_row = eligible.sample(n=1, replace=True, random_state=seed + i + 1).iloc[0]
        provider_info = provider_lookup[source_row["provider_id"]]
        row = _base_output_row(
            source_row,
            provider_info,
            "ANOMALY",
            f"anomaly_{profile}",
            "Suspicious" if profile == "borderline" else "High Risk",
            rng,
            used_txn_ids,
            used_patient_ids,
        )

        median_billing = max(_safe_float(provider_info.get("median_billing"), row["billing_amount"]), 100.0)
        q75_billing = max(_safe_float(provider_info.get("q75_billing"), median_billing), median_billing)
        anomaly_type = rng.choice(["amount_spike", "rapid_repeat", "phantom_stay", "clinical_mismatch"])
        anchor_time = pd.to_datetime(provider_info["last_service_start"])

        if profile == "borderline":
            mild_multiplier = rng.uniform(4.00, 6.50)
            high_multiplier = rng.uniform(6.00, 10.00)
            minimum_amount = 8000.0
            stay_range = (8, 18)
        else:
            mild_multiplier = rng.uniform(8.00, 12.00)
            high_multiplier = rng.uniform(12.00, 20.00)
            minimum_amount = 15000.0
            stay_range = (20, 45)

        row["service_start"] = anchor_time + timedelta(minutes=rng.randint(15, 90))
        row["service_end"] = row["service_start"]
        row["admission_date"] = pd.to_datetime(row["service_start"]).date()

        if anomaly_type == "amount_spike":
            row["billing_amount"] = round(max(row["billing_amount"], median_billing, q75_billing, minimum_amount) * high_multiplier, 2)
            row["deductible_context"] = max(row["deductible_context"], 4096.0 if profile == "extreme" else 2048.0)
            row["coinsurance_context"] = max(row["coinsurance_context"], 512.0 if profile == "extreme" else 256.0)
        elif anomaly_type == "rapid_repeat":
            row["claim_type"] = "Outpatient"
            row["billing_amount"] = round(max(row["billing_amount"], median_billing, q75_billing, minimum_amount) * mild_multiplier, 2)
            row["deductible_context"] = max(row["deductible_context"], 1024.0 if profile == "borderline" else 2048.0)
            row["coinsurance_context"] = max(row["coinsurance_context"], 128.0 if profile == "borderline" else 256.0)
        elif anomaly_type == "clinical_mismatch":
            row["claim_type"] = "Outpatient"
            row["service_end"] = row["service_start"]
            row["diagnosis_context"] = rng.choice(["V222", "V700", "4011", "4280"])
            row["primary_procedure"] = rng.choice(["3340", "44140", "8154", "3722"])
            row["treatment_group"] = rng.choice(["999", "998", "997", "089"])
            row["billing_amount"] = round(max(row["billing_amount"], median_billing, q75_billing, minimum_amount) * high_multiplier, 2)
            row["deductible_context"] = max(row["deductible_context"], 1024.0 if profile == "borderline" else 2048.0)
            row["coinsurance_context"] = max(row["coinsurance_context"], 128.0 if profile == "borderline" else 256.0)
        else:
            row["claim_type"] = "Inpatient"
            stay_days = rng.randint(*stay_range)
            row["service_end"] = pd.to_datetime(row["service_start"]) + timedelta(days=stay_days)
            row["deductible_context"] = max(row["deductible_context"], 4096.0 if profile == "extreme" else 2048.0)
            row["coinsurance_context"] = max(row["coinsurance_context"], 512.0 if profile == "extreme" else 256.0)
            row["billing_amount"] = round(max(row["billing_amount"], median_billing, q75_billing, minimum_amount) * high_multiplier, 2)

        row["has_diabetes"] = 1
        row["has_chf"] = 1
        row["has_cancer"] = 1 if profile == "extreme" else rng.choice([0, 1])
        row["has_copd"] = 1 if profile == "extreme" else rng.choice([0, 1])
        if anomaly_type != "clinical_mismatch":
            row["diagnosis_context"] = rng.choice(["V222", "V700", "4011", "4280"])
            row["primary_procedure"] = rng.choice(["3340", "44140", "8154", "3722"])
            row["treatment_group"] = rng.choice(["999", "998", "997", "089"])

        rows.append(_finalize_dates(row))

    return rows


def generate_mixed_eval_data(normal_rows=50, anomaly_rows=50, seed=123, output_path=OUTPUT_PATH):
    df = _load_raw_transactions()
    provider_context = _build_provider_context(df)
    if provider_context.empty:
        raise RuntimeError("Not enough provider history in Healthcare_transactions to build evaluation data.")

    normal_pool = _build_normal_pool(df, provider_context)
    if normal_pool.empty:
        raise RuntimeError("Could not build a raw normal pool from Healthcare_transactions.")

    clear_normal_rows = normal_rows // 2
    borderline_normal_rows = normal_rows - clear_normal_rows
    borderline_anomaly_rows = anomaly_rows // 2
    extreme_anomaly_rows = anomaly_rows - borderline_anomaly_rows

    rows = []
    rows.extend(_make_normal_rows(normal_pool, provider_context, clear_normal_rows, "clear", seed))
    rows.extend(_make_normal_rows(normal_pool, provider_context, borderline_normal_rows, "borderline", seed + 11))
    rows.extend(_make_anomaly_rows(normal_pool, provider_context, borderline_anomaly_rows, "borderline", seed + 21))
    rows.extend(_make_anomaly_rows(normal_pool, provider_context, extreme_anomaly_rows, "extreme", seed + 31))

    mixed_df = _clean_output_dataframe(pd.DataFrame(rows, columns=OUTPUT_COLUMNS))
    mixed_df = mixed_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mixed_df.to_csv(output_path, index=False)
    print(
        f"Generated {len(mixed_df)} mixed evaluation rows at {output_path} "
        f"({clear_normal_rows} clear normal + {borderline_normal_rows} borderline normal + "
        f"{borderline_anomaly_rows} suspicious anomaly + {extreme_anomaly_rows} high-risk anomaly). "
        "Source: Healthcare_transactions only. No model scoring was used during generation."
    )
    return mixed_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a mixed evaluation CSV from Healthcare_transactions only."
    )
    parser.add_argument("--normal-rows", type=int, default=50, help="Number of normal rows to generate.")
    parser.add_argument("--anomaly-rows", type=int, default=50, help="Number of anomaly rows to generate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible generation.")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_mixed_eval_data(
        normal_rows=args.normal_rows,
        anomaly_rows=args.anomaly_rows,
        seed=args.seed,
        output_path=args.output,
    )
