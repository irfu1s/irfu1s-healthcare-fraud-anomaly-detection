import argparse
import os
import sys
import time

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_DIR = os.path.join(BASE_DIR, "tests")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from src.model_inference import run_inference
from generate_mixed_eval_data import OUTPUT_PATH as MIXED_OUTPUT_PATH, generate_mixed_eval_data


def normalize_label(value):
    text = str(value or "").strip().upper()
    if "ANOM" in text or "FRAUD" in text:
        return "ANOMALY"
    if "NORMAL" in text:
        return "NORMAL"
    return "UNKNOWN"


def predicted_label(result):
    return "ANOMALY" if str(result.get("is_normal", "Yes")).strip().lower() == "no" else "NORMAL"


def ensure_test_files():
    if not os.path.exists(MIXED_OUTPUT_PATH):
        generate_mixed_eval_data()


def run_batch_test(csv_path=MIXED_OUTPUT_PATH):
    if csv_path == MIXED_OUTPUT_PATH:
        ensure_test_files()

    df_test = pd.read_csv(csv_path)

    results = []
    start_time = time.time()

    print(f"Running inference on {len(df_test)} rows from {csv_path}...")

    for index, row in df_test.iterrows():
        transaction = row.to_dict()
        actual = normalize_label(transaction.pop("Label", "UNKNOWN"))
        scenario_profile = transaction.get("scenario_profile", "unknown")
        expected_zone = transaction.get("expected_zone", "UNKNOWN")

        try:
            result = run_inference(transaction)
            predicted = predicted_label(result)
            alert_zone = result.get("alert_zone", "UNKNOWN")
            results.append(
                {
                    "transaction_id": transaction.get("transaction_id"),
                    "provider_id": transaction.get("provider_id"),
                    "scenario_profile": scenario_profile,
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "expected_zone": expected_zone,
                    "alert_zone": alert_zone,
                    "is_correct": actual == predicted if actual != "UNKNOWN" else None,
                    "zone_is_correct": expected_zone == alert_zone if expected_zone != "UNKNOWN" else None,
                    "anomaly_score": result.get("anomaly_score"),
                    "reason": result.get("reason"),
                }
            )
        except Exception as exc:
            print(f"Error on row {index} (transaction_id={transaction.get('transaction_id')}): {exc}")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results were generated. Check model loading or inference input handling.")
        return results_df

    comparable = results_df["is_correct"].dropna()
    accuracy = float(comparable.mean() * 100) if not comparable.empty else 0.0
    comparable_zone = results_df["zone_is_correct"].dropna()
    zone_accuracy = float(comparable_zone.mean() * 100) if not comparable_zone.empty else 0.0
    zone_summary = (
        results_df.groupby(["scenario_profile", "alert_zone"])
        .size()
        .reset_index(name="count")
        .sort_values(["scenario_profile", "alert_zone"])
    )

    output_file = os.path.join(BASE_DIR, "data", "processed", "batch_inference_results.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"\nBatch inference complete in {round(time.time() - start_time, 2)} seconds.")
    print(f"Binary Accuracy: {accuracy:.2f}%")
    print(f"Zone Accuracy: {zone_accuracy:.2f}%")
    print("\nActual vs Predicted:")
    print(
        results_df[
            [
                "transaction_id",
                "provider_id",
                "scenario_profile",
                "actual_label",
                "predicted_label",
                "expected_zone",
                "alert_zone",
                "is_correct",
                "zone_is_correct",
                "anomaly_score",
            ]
        ].to_string(index=False)
    )
    print("\nZone summary:")
    print(zone_summary.to_string(index=False))

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference on a generated mixed CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default=MIXED_OUTPUT_PATH,
        help="Path to the CSV file used for batch inference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch_test(csv_path=args.csv)
