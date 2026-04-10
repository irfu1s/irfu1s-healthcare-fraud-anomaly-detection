import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_DIR = os.path.join(BASE_DIR, "tests")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from generate_demo_data import generate_normal_data
from genrate_anomaly import generate_anomaly_data

OUTPUT_PATH = os.path.join(SAMPLES_DIR, "mixed_eval_data.csv")


def _clean_mixed_output(df):
    df = df.copy()
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

    for col, default in text_defaults.items():
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA, "NaT": pd.NA})
                .fillna(default)
            )
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
    return df


def generate_mixed_eval_data(normal_rows=50, anomaly_rows=50, seed=123, output_path=OUTPUT_PATH):
    clear_normal_rows = normal_rows // 2
    borderline_normal_rows = normal_rows - clear_normal_rows
    borderline_anomaly_rows = anomaly_rows // 2
    extreme_anomaly_rows = anomaly_rows - borderline_anomaly_rows

    normal_clear_df = generate_normal_data(
        row_count=clear_normal_rows,
        seed=seed,
        output_path=None,
        profile="clear",
    )
    normal_borderline_df = generate_normal_data(
        row_count=borderline_normal_rows,
        seed=seed + 11,
        output_path=None,
        profile="borderline",
    )
    anomaly_borderline_df = generate_anomaly_data(
        row_count=borderline_anomaly_rows,
        seed=seed + 21,
        output_path=None,
        profile="borderline",
    )
    anomaly_extreme_df = generate_anomaly_data(
        row_count=extreme_anomaly_rows,
        seed=seed + 31,
        output_path=None,
        profile="extreme",
    )

    mixed_df = (
        pd.concat(
            [
                normal_clear_df,
                normal_borderline_df,
                anomaly_borderline_df,
                anomaly_extreme_df,
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    mixed_df = _clean_mixed_output(mixed_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mixed_df.to_csv(output_path, index=False)
    print(
        f"Generated {len(mixed_df)} mixed evaluation rows at {output_path} "
        f"({clear_normal_rows} clear normal + {borderline_normal_rows} borderline normal + "
        f"{borderline_anomaly_rows} suspicious anomaly + {extreme_anomaly_rows} high-risk anomaly)"
    )
    return mixed_df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a mixed normal/anomaly evaluation CSV.")
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
