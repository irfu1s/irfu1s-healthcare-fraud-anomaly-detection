from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from textwrap import shorten
import os
import re

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="Healthcare Anomaly Detection",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
BATCH_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "batch_inference_results.csv"
BATCH_MAX_WORKERS = max(1, int(os.getenv("BATCH_MAX_WORKERS", "3")))
BATCH_ROW_TIMEOUT = max(30, int(os.getenv("BATCH_ROW_TIMEOUT", "180")))
BATCH_ROW_RETRIES = max(0, int(os.getenv("BATCH_ROW_RETRIES", "1")))

TRANSACTION_ID_PATTERN = re.compile(r"^\d{14}$")
PATIENT_ID_PATTERN = re.compile(r"^[A-Z0-9]{16}$")
DEFAULT_SERVICE_DATE = date(2008, 12, 15)

ZONE_META = {
    "Normal": {
        "action": "Auto-clear candidate",
        "summary": "Claim behavior is within the expected provider pattern.",
    },
    "Suspicious": {
        "action": "Manual review required",
        "summary": "The claim sits near the anomaly boundary and should be reviewed.",
    },
    "High Risk": {
        "action": "Escalate to audit",
        "summary": "The claim strongly deviates from learned provider behavior.",
    },
}


def get_backend_url() -> str:
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000/evaluate_transaction").strip()


def get_batch_backend_url() -> str:
    single_url = get_backend_url()
    if single_url.endswith("/evaluate_transaction"):
        return single_url.rsplit("/", 1)[0] + "/evaluate_batch"
    return os.getenv("BATCH_BACKEND_URL", "http://127.0.0.1:8000/evaluate_batch").strip()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main-app-title {
                text-align: center;
                font-size: 2.25rem;
                font-weight: 800;
                color: #38bdf8;
                margin-bottom: 0.25rem;
            }
            .main-app-subtitle {
                text-align: center;
                font-size: 14px;
                margin-bottom: 1.6rem;
                color: inherit;
            }
            .page-section-title {
                text-align: center;
                font-size: 18px;
                font-weight: 800;
                margin-top: 0.4rem;
                margin-bottom: 0.2rem;
            }
            .page-section-copy {
                text-align: center;
                font-size: 14px;
                margin-bottom: 1.2rem;
                color: inherit;
            }
            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stCaptionContainer"],
            label,
            small {
                font-size: 14px !important;
            }
            div[data-testid="stMarkdownContainer"] h3,
            div[data-testid="stMarkdownContainer"] h4 {
                font-size: 18px !important;
                font-weight: 800 !important;
            }
            div[data-testid="stMetricLabel"] p {
                font-size: 14px !important;
            }
            div[data-testid="stMetricValue"] {
                font-size: 16px !important;
            }
            div[data-testid="stFileUploader"] {
                max-width: 440px;
                margin: 0 auto;
            }
            section[data-testid="stFileUploaderDropzone"] {
                background: rgba(255, 255, 255, 0.04);
                border: 2px dashed rgba(148, 163, 184, 0.75);
                border-radius: 22px;
                min-height: 250px;
                padding: 1.5rem 1.2rem;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            section[data-testid="stFileUploaderDropzone"]:hover {
                border-color: #0f766e;
                background: rgba(15, 118, 110, 0.08);
            }
            section[data-testid="stFileUploaderDropzone"] > div {
                width: 100%;
            }
            section[data-testid="stFileUploaderDropzone"] > div > div {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 0.75rem;
                width: 100%;
            }
            section[data-testid="stFileUploaderDropzoneInstructions"] > div {
                color: inherit;
                font-size: 1.02rem;
                font-weight: 600;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            section[data-testid="stFileUploaderDropzoneInstructions"] small {
                color: rgba(255, 255, 255, 0.72);
                font-size: 0.88rem;
            }
            section[data-testid="stFileUploaderDropzone"] svg {
                width: 52px;
                height: 52px;
                color: #0f766e;
                margin-bottom: 0.65rem;
            }
            div[data-testid="stFileUploader"] button {
                border-radius: 12px;
                font-weight: 700;
                margin-top: 0.6rem;
                margin-left: auto;
                margin-right: auto;
            }
            div[data-testid="stDataFrame"] div[role="columnheader"] {
                font-size: 12px !important;
                font-weight: 700 !important;
            }
            div[data-testid="stDataFrame"] div[role="gridcell"] {
                font-size: 12px !important;
            }
            .zone-callout {
                border-radius: 12px;
                padding: 0.95rem 1rem;
                margin: 0.85rem 0;
                border: 1px solid transparent;
            }
            .zone-callout-title {
                font-size: 16px;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }
            .zone-callout-copy {
                font-size: 14px;
                line-height: 1.45;
            }
            .zone-normal {
                background: rgba(34, 197, 94, 0.18);
                border-color: rgba(34, 197, 94, 0.45);
                color: #bbf7d0;
            }
            .zone-suspicious {
                background: rgba(245, 158, 11, 0.18);
                border-color: rgba(250, 204, 21, 0.45);
                color: #fde68a;
            }
            .zone-high-risk {
                background: rgba(239, 68, 68, 0.18);
                border-color: rgba(248, 113, 113, 0.5);
                color: #fecaca;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "single_result": None,
        "single_payload": None,
        "single_error": None,
        "batch_results": None,
        "batch_signature": None,
        "batch_error": None,
        "batch_failures": 0,
        "batch_workers": BATCH_MAX_WORKERS,
        "batch_running": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> str:
    with st.sidebar:
        st.title("Sentinel AI")
        st.caption("Healthcare fraud screening dashboard")
        st.markdown("**Model:** Isolation Forest + LSTM")
        st.markdown("**Threshold:** 92.5th percentile")
        st.markdown("**Zones:** Normal, Suspicious, High Risk")
        st.divider()
        return st.radio("Navigation", ["Single Claim Analysis", "Batch Audit Dashboard"])


def render_header() -> None:
    st.markdown('<div class="main-app-title">Healthcare Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-app-subtitle">Real-time claim risk evaluation using Isolation Forest and LSTM</div>',
        unsafe_allow_html=True,
    )


def parse_money(value: str, label: str, errors: list[str]) -> float:
    text = str(value).replace(",", "").replace("$", "").strip()
    if not text:
        errors.append(label)
        return 0.0
    try:
        return float(text)
    except ValueError:
        errors.append(f"{label} as a valid number")
        return 0.0


def normalize_claim_type(value: object) -> str:
    text = str(value or "Outpatient").strip()
    lowered = text.lower()
    if "inpatient" in lowered:
        return "Inpatient"
    if "outpatient" in lowered:
        return "Outpatient"
    return text or "Outpatient"


def format_datetime_value(value: object, fallback_time: str = "09:00:00") -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0:
        hour, minute, second = [int(part) for part in fallback_time.split(":")]
        timestamp = timestamp.replace(hour=hour, minute=minute, second=second)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_date_value(value: object) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def clean_cell_value(value: object) -> object | None:
    if is_missing_value(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "nan", "none", "null", "nat"}:
            return None
        return text
    return value


def first_present(cleaned: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        value = clean_cell_value(cleaned.get(key))
        if value is not None:
            return value
    return None


def coerce_float_field(
    cleaned: dict[str, object],
    *keys: str,
    default: float = 0.0,
    required: bool = False,
    label: str = "value",
) -> float:
    value = first_present(cleaned, *keys)
    if value is None:
        if required:
            raise ValueError(f"Missing {label}")
        return default
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        if required:
            raise ValueError(f"Invalid {label}")
        return default


def coerce_int_field(
    cleaned: dict[str, object],
    *keys: str,
    default: int = 0,
    required: bool = False,
    label: str = "value",
) -> int:
    value = first_present(cleaned, *keys)
    if value is None:
        if required:
            raise ValueError(f"Missing {label}")
        return default
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        if required:
            raise ValueError(f"Invalid {label}")
        return default


def coerce_text_field(
    cleaned: dict[str, object],
    *keys: str,
    default: str = "0",
    uppercase: bool = False,
    required: bool = False,
    label: str = "value",
) -> str:
    value = first_present(cleaned, *keys)
    if value is None:
        if required:
            raise ValueError(f"Missing {label}")
        text = default
    else:
        text = str(value).strip()
    if uppercase:
        text = text.upper()
    return text


def coerce_binary_flag(value: object) -> int:
    value = clean_cell_value(value)
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "present"}:
            return 1
        if text in {"0", "false", "no", "n", ""}:
            return 0
    try:
        return 1 if float(value) > 0 else 0
    except (TypeError, ValueError):
        return 0


def normalize_true_label(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    upper_text = text.upper()
    if "ANOM" in upper_text or "FRAUD" in upper_text:
        return "ANOMALY"
    if "NORMAL" in upper_text:
        return "NORMAL"
    return upper_text


def predicted_label_from_result(result: dict[str, object]) -> str:
    return "ANOMALY" if str(result.get("is_normal", "Yes")).strip().lower() == "no" else "NORMAL"


def coerce_transaction_id(raw_value: object, strict_14_digits: bool = False) -> int:
    text = str(raw_value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if strict_14_digits and not TRANSACTION_ID_PATTERN.fullmatch(text):
        raise ValueError("Transaction ID must contain exactly 14 numeric digits.")
    if not text.isdigit():
        raise ValueError("Transaction ID must be numeric.")
    return int(text)


def call_backend(
    payload: dict[str, object],
    include_explainability: bool = True,
    timeout_seconds: int = 30,
) -> dict[str, object]:
    response = requests.post(
        get_backend_url(),
        json=payload,
        params={"include_explainability": str(include_explainability).lower()},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def call_batch_backend(payloads: list[dict[str, object]]) -> dict[str, object]:
    timeout_seconds = max(120, min(1800, len(payloads) * 20))
    response = requests.post(
        get_batch_backend_url(),
        json={"transactions": payloads},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def build_single_claim_payload(form_values: dict[str, object]) -> tuple[dict[str, object] | None, list[str]]:
    errors: list[str] = []

    transaction_text = str(form_values["transaction_id"]).strip()
    patient_id = str(form_values["patient_id"]).strip().upper()
    provider_id = str(form_values["provider_id"]).strip().upper()
    diagnosis_code = str(form_values["diagnosis_code"]).strip().upper()
    claim_type = normalize_claim_type(form_values["claim_type"])

    billing_amount = parse_money(str(form_values["billing_amount"]), "billing amount", errors)
    deductible = parse_money(str(form_values["deductible"]), "deductible", errors)
    coinsurance = parse_money(str(form_values["coinsurance"]), "coinsurance", errors)

    if not TRANSACTION_ID_PATTERN.fullmatch(transaction_text):
        errors.append("14-digit transaction ID")
    if not PATIENT_ID_PATTERN.fullmatch(patient_id):
        errors.append("16-character alphanumeric patient ID")
    if not provider_id:
        errors.append("provider ID")
    if not diagnosis_code:
        errors.append("diagnosis code")
    if billing_amount <= 0:
        errors.append("billing amount greater than 0")
    if int(form_values["patient_age"]) <= 0:
        errors.append("patient age greater than 0")

    service_start_date = form_values["service_date"]
    service_end_date = form_values["service_end_date"] if claim_type == "Inpatient" else service_start_date
    if claim_type == "Inpatient" and service_end_date < service_start_date:
        errors.append("end of service date on or after the start date")

    procedure_code = str(form_values["primary_procedure"]).strip() if claim_type == "Inpatient" else "0"
    treatment_code = str(form_values["treatment_group"]).strip() if claim_type == "Inpatient" else "0"
    if claim_type == "Inpatient":
        if not procedure_code:
            errors.append("procedure code")
        if not treatment_code:
            errors.append("treatment code")

    if errors:
        return None, errors

    service_start = f"{service_start_date.isoformat()} 09:00:00"
    service_end = f"{service_end_date.isoformat()} {'11:00:00' if claim_type == 'Inpatient' else '10:00:00'}"

    payload = {
        "transaction_id": coerce_transaction_id(transaction_text, strict_14_digits=True),
        "patient_id": patient_id,
        "provider_id": provider_id,
        "claim_type": claim_type,
        "billing_amount": billing_amount,
        "deductible_context": deductible,
        "coinsurance_context": coinsurance,
        "service_start": service_start,
        "service_end": service_end,
        "patient_age": int(form_values["patient_age"]),
        "diagnosis_context": diagnosis_code,
        "has_diabetes": 1 if form_values["has_diabetes"] else 0,
        "has_chf": 1 if form_values["has_chf"] else 0,
        "has_cancer": 1 if form_values["has_cancer"] else 0,
        "has_copd": 1 if form_values["has_copd"] else 0,
        "primary_procedure": procedure_code,
        "treatment_group": treatment_code,
        "admission_date": service_start_date.isoformat(),
        "gender_context": 1,
        "demographic_context": 1,
        "primary_service": "0",
    }
    return payload, []


def build_batch_payload(row_dict: dict[str, object]) -> dict[str, object]:
    cleaned = {key: clean_cell_value(value) for key, value in row_dict.items()}

    claim_type = normalize_claim_type(cleaned.get("claim_type"))
    service_start = (
        format_datetime_value(cleaned.get("service_start"), "09:00:00")
        or format_datetime_value(cleaned.get("date_of_service"), "09:00:00")
        or format_datetime_value(cleaned.get("service_date"), "09:00:00")
    )
    if not service_start:
        raise ValueError("Missing service_start/date_of_service")

    service_end = format_datetime_value(cleaned.get("service_end"), "10:00:00")
    if not service_end:
        start_dt = pd.to_datetime(service_start)
        if claim_type == "Inpatient":
            end_dt = (start_dt + timedelta(days=2)).replace(hour=11, minute=0, second=0)
            service_end = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            service_end = start_dt.replace(hour=10, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")

    admission_date = format_date_value(cleaned.get("admission_date")) or format_date_value(service_start)
    diagnosis_context = coerce_text_field(
        cleaned,
        "diagnosis_context",
        "diagnosis_code",
        default="UNKNOWN",
        uppercase=True,
    )
    primary_procedure = coerce_text_field(
        cleaned,
        "primary_procedure",
        "procedure_code",
        default="1000" if claim_type == "Inpatient" else "0",
    )
    treatment_group = coerce_text_field(
        cleaned,
        "treatment_group",
        "treatment_code",
        default="100" if claim_type == "Inpatient" else "0",
    )

    patient_id = coerce_text_field(cleaned, "patient_id", default="UNKNOWN", uppercase=True)
    provider_id = coerce_text_field(cleaned, "provider_id", default="", uppercase=True)
    if not provider_id:
        raise ValueError("Missing provider_id")

    billing_amount = coerce_float_field(
        cleaned,
        "billing_amount",
        "claim_amount",
        default=0.0,
        required=True,
        label="billing_amount",
    )
    if billing_amount <= 0:
        raise ValueError("Billing amount must be greater than 0")

    return {
        "transaction_id": coerce_transaction_id(cleaned.get("transaction_id")),
        "patient_id": patient_id,
        "provider_id": provider_id,
        "claim_type": claim_type,
        "billing_amount": billing_amount,
        "deductible_context": coerce_float_field(
            cleaned,
            "deductible_context",
            "deductible",
            default=0.0,
        ),
        "coinsurance_context": coerce_float_field(
            cleaned,
            "coinsurance_context",
            "coinsurance",
            default=0.0,
        ),
        "service_start": service_start,
        "service_end": service_end,
        "patient_age": coerce_int_field(cleaned, "patient_age", "age", default=45),
        "diagnosis_context": diagnosis_context,
        "has_diabetes": coerce_binary_flag(cleaned.get("has_diabetes")),
        "has_chf": coerce_binary_flag(cleaned.get("has_chf")),
        "has_cancer": coerce_binary_flag(cleaned.get("has_cancer")),
        "has_copd": coerce_binary_flag(cleaned.get("has_copd")),
        "primary_procedure": primary_procedure,
        "treatment_group": treatment_group,
        "admission_date": admission_date,
        "gender_context": coerce_int_field(cleaned, "gender_context", default=1),
        "demographic_context": coerce_int_field(cleaned, "demographic_context", default=1),
        "primary_service": coerce_text_field(cleaned, "primary_service", default="0"),
    }


def build_shap_impact_chart(result: dict[str, object]) -> go.Figure | None:
    shap_data = result.get("shap_data") or {}
    features = shap_data.get("features") or []
    contributions = shap_data.get("contributions") or []
    values = shap_data.get("feature_values") or []

    if not features or not contributions or len(features) != len(contributions):
        return None

    if len(values) != len(features):
        values = [""] * len(features)

    ranked = sorted(
        zip(features, contributions, values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:8]

    y_values = [item[0] for item in ranked][::-1]
    x_values = [float(item[1]) for item in ranked][::-1]
    observed = [str(item[2]) for item in ranked][::-1]
    colors = ["#d92d20" if value > 0 else "#15803d" for value in x_values]

    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=y_values,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{value:+.3f}" for value in x_values],
            textposition="outside",
            customdata=observed,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Observed value: %{customdata}<br>"
                "Impact: %{x:+.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Impact on anomaly score",
        yaxis_title="",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8", line_width=1.2)
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def get_driver_bullets(result: dict[str, object]) -> list[str]:
    drivers = result.get("top_drivers") or []
    if drivers:
        return [str(item) for item in drivers[:4]]
    reason = str(result.get("reason") or "").strip()
    return [reason] if reason else []


def render_zone_callout(zone: str, summary: str) -> None:
    zone_class = {
        "Normal": "zone-normal",
        "Suspicious": "zone-suspicious",
        "High Risk": "zone-high-risk",
    }.get(zone, "zone-normal")
    st.markdown(
        f"""
        <div class="zone-callout {zone_class}">
            <div class="zone-callout-title">{zone}</div>
            <div class="zone-callout-copy">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_panel(result: dict[str, object] | None, payload: dict[str, object] | None) -> None:
    st.subheader("AI Result Panel")

    with st.container(border=True):
        if not result:
            st.metric("Anomaly Score", "--")
            st.info("Run AI Risk Analysis to display the model score, risk zone, explanation, and feature impact.")
            return

        zone = str(result.get("alert_zone") or "Normal")
        meta = ZONE_META.get(zone, ZONE_META["Normal"])
        score = float(result.get("anomaly_score") or 0.0)
        binary_decision = "Anomaly" if str(result.get("is_normal")) == "No" else "Normal"

        top_col, top_right = st.columns([1, 1])
        with top_col:
            st.metric("Anomaly Score", f"{score:.4f}")
        with top_right:
            st.metric("Risk Category", zone)

        render_zone_callout(zone, meta["summary"])

        metric_cols = st.columns(3)
        metric_cols[0].metric("Binary Decision", binary_decision)
        metric_cols[1].metric("Action", meta["action"])
        metric_cols[2].metric("Provider ID", str((payload or {}).get("provider_id", result.get("provider_id", "N/A"))))

    with st.container(border=True):
        st.markdown("#### Anomaly Explanation")
        reason_text = str(result.get("reason") or "").strip()
        if reason_text:
            st.write(reason_text)
        bullets = get_driver_bullets(result)
        if bullets:
            for bullet in bullets:
                st.markdown(f"- {bullet}")
        else:
            st.caption("No explanation text was returned by the backend.")

    with st.container(border=True):
        st.markdown("#### Feature Contributions")
        chart = build_shap_impact_chart(result)
        if chart is None:
            st.info("Feature contribution data is not available for this result.")
        else:
            st.plotly_chart(chart, width="stretch")


def run_single_claim_form() -> None:
    left_col, right_col = st.columns([1.08, 1.02], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("Claim Details")
            st.caption("Enter the core fields below and run the live AI risk analysis.")

            st.markdown("#### Identity")
            id_left, id_right = st.columns(2)
            with id_left:
                transaction_id = st.text_input(
                    "Transaction ID",
                    placeholder="14 numeric digits",
                    help="Example: 89509112315050",
                )
            with id_right:
                patient_id = st.text_input(
                    "Patient ID",
                    placeholder="16 alphanumeric characters",
                    help="Example: B55D152A96663E01",
                )

            type_left, type_right = st.columns(2)
            with type_left:
                provider_id = st.text_input(
                    "Provider ID",
                    placeholder="Existing provider code",
                    help="Example: 2300YM",
                )
            with type_right:
                claim_type = st.radio(
                    "Claim Type",
                    ["Outpatient", "Inpatient"],
                    horizontal=True,
                    key="single_claim_type",
                )

            st.markdown("#### Financial")
            finance_left, finance_mid, finance_right = st.columns(3)
            with finance_left:
                billing_amount = st.text_input("Billing Amount ($)", placeholder="e.g. 14802.77")
            with finance_mid:
                deductible = st.text_input("Deductible ($)", placeholder="e.g. 0.00")
            with finance_right:
                coinsurance = st.text_input("Coinsurance ($)", placeholder="e.g. 40.00")

            st.markdown("#### Clinical Context")
            clinical_left, clinical_right = st.columns([0.9, 1.1])
            with clinical_left:
                patient_age = st.number_input(
                    "Patient Age",
                    min_value=0,
                    max_value=120,
                    value=45,
                    step=1,
                )
                diagnosis_code = st.text_input("Diagnosis Code", placeholder="e.g. V222")
            with clinical_right:
                st.markdown("**Chronic Conditions**")
                cond_left, cond_right = st.columns(2)
                with cond_left:
                    has_diabetes = st.checkbox("Diabetes")
                    has_chf = st.checkbox("CHF / Heart Failure")
                with cond_right:
                    has_cancer = st.checkbox("Cancer")
                    has_copd = st.checkbox("COPD")

            st.markdown("#### Visit Details")
            service_date = st.date_input(
                "Date of Service",
                value=DEFAULT_SERVICE_DATE,
                format="DD/MM/YYYY",
            )

            service_end_date = service_date
            treatment_group = "0"
            primary_procedure = "0"
            if claim_type == "Inpatient":
                inpatient_left, inpatient_mid, inpatient_right = st.columns(3)
                with inpatient_left:
                    service_end_date = st.date_input(
                        "End of Service Date",
                        value=service_date + timedelta(days=2),
                        min_value=service_date,
                        format="DD/MM/YYYY",
                        key="inpatient_end_date",
                    )
                with inpatient_mid:
                    treatment_group = st.text_input(
                        "Treatment Code",
                        placeholder="e.g. 100",
                        key="inpatient_treatment_code",
                    )
                with inpatient_right:
                    primary_procedure = st.text_input(
                        "Procedure Code",
                        placeholder="e.g. 1000",
                        key="inpatient_procedure_code",
                    )

            submitted = st.button("Run AI Risk Analysis", width="stretch", type="primary")

        if st.session_state["single_error"]:
            st.error(st.session_state["single_error"])

        if submitted:
            payload, errors = build_single_claim_payload(
                {
                    "transaction_id": transaction_id,
                    "patient_id": patient_id,
                    "provider_id": provider_id,
                    "claim_type": claim_type,
                    "billing_amount": billing_amount,
                    "deductible": deductible,
                    "coinsurance": coinsurance,
                    "patient_age": patient_age,
                    "diagnosis_code": diagnosis_code,
                    "has_diabetes": has_diabetes,
                    "has_chf": has_chf,
                    "has_cancer": has_cancer,
                    "has_copd": has_copd,
                    "service_date": service_date,
                    "service_end_date": service_end_date,
                    "treatment_group": treatment_group,
                    "primary_procedure": primary_procedure,
                }
            )

            if errors:
                st.session_state["single_result"] = None
                st.session_state["single_payload"] = None
                st.session_state["single_error"] = "Please provide: " + ", ".join(errors)
            else:
                try:
                    with st.spinner("Running live AI risk analysis..."):
                        result = call_backend(payload)
                except requests.exceptions.ConnectionError:
                    st.session_state["single_result"] = None
                    st.session_state["single_payload"] = None
                    st.session_state["single_error"] = "Connection failed. Start the FastAPI backend with `uvicorn API.api:app --reload`."
                except requests.exceptions.Timeout:
                    st.session_state["single_result"] = None
                    st.session_state["single_payload"] = None
                    st.session_state["single_error"] = "The backend timed out while loading or scoring the claim. Please try again."
                except requests.exceptions.RequestException as exc:
                    error_text = getattr(exc.response, "text", str(exc))
                    st.session_state["single_result"] = None
                    st.session_state["single_payload"] = None
                    st.session_state["single_error"] = f"API error: {error_text}"
                else:
                    st.session_state["single_result"] = result
                    st.session_state["single_payload"] = payload
                    st.session_state["single_error"] = None

    with right_col:
        render_result_panel(st.session_state["single_result"], st.session_state["single_payload"])


def get_uploaded_signature(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}:{uploaded_file.size}"


def build_batch_result_row(
    index: int,
    row_dict: dict[str, object],
    payload: dict[str, object] | None,
    result: dict[str, object],
) -> dict[str, object]:
    combined = {
        "_row_order": index,
        "transaction_id": result.get(
            "transaction_id",
            payload.get("transaction_id") if payload else row_dict.get("transaction_id"),
        ),
        "provider_id": result.get(
            "provider_id",
            payload.get("provider_id") if payload else row_dict.get("provider_id"),
        ),
        "patient_id": payload.get("patient_id") if payload else row_dict.get("patient_id"),
        "claim_type": payload.get("claim_type") if payload else row_dict.get("claim_type"),
        "anomaly_score": result.get("anomaly_score"),
        "alert_zone": result.get("alert_zone"),
        "is_normal": result.get("is_normal"),
        "predicted_label": predicted_label_from_result(result),
        "reason": result.get("reason"),
        "batch_status": result.get("batch_status", "Processed"),
    }

    true_label = (
        normalize_true_label(row_dict.get("true_label"))
        or normalize_true_label(row_dict.get("actual_label"))
        or normalize_true_label(row_dict.get("Label"))
        or normalize_true_label(row_dict.get("label"))
    )
    if true_label:
        combined["true_label"] = true_label

    for optional_column in ["scenario_profile"]:
        if optional_column in row_dict and pd.notna(row_dict[optional_column]):
            combined[optional_column] = row_dict[optional_column]
    return combined


def run_batch_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    total_rows = len(df)
    results: list[dict[str, object]] = []

    progress_bar = st.progress(0)
    status = st.empty()

    payloads: list[dict[str, object]] = []
    valid_rows: list[tuple[int, dict[str, object], dict[str, object]]] = []

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        row_dict = row.to_dict()
        try:
            payload = build_batch_payload(row_dict)
        except Exception as exc:
            results.append(
                build_batch_result_row(
                    index,
                    row_dict,
                    None,
                    {
                        "transaction_id": row_dict.get("transaction_id"),
                        "provider_id": row_dict.get("provider_id"),
                        "anomaly_score": None,
                        "alert_zone": "Processing Failed",
                        "is_normal": "Error",
                        "reason": f"Input validation failed: {exc}",
                        "batch_status": "Failed",
                    },
                )
            )
            continue

        payloads.append(payload)
        valid_rows.append((index, row_dict, payload))

        if total_rows:
            progress_bar.progress(min(index / total_rows, 0.15))
            status.caption(f"Validated {index} of {total_rows} uploaded rows")

    worker_count = min(BATCH_MAX_WORKERS, len(valid_rows)) if valid_rows else 1
    if valid_rows:
        completed = 0
        status.caption(f"Submitting {len(valid_rows)} validated claims with {worker_count} parallel API requests...")
        progress_bar.progress(0.15)

        def score_one(row_info: tuple[int, dict[str, object], dict[str, object]]) -> dict[str, object]:
            index, row_dict, payload = row_info
            try:
                last_error = None
                for attempt in range(BATCH_ROW_RETRIES + 1):
                    try:
                        result = call_backend(
                            payload,
                            include_explainability=False,
                            timeout_seconds=BATCH_ROW_TIMEOUT,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    raise last_error if last_error else RuntimeError("Unknown batch API failure.")

                result["batch_status"] = result.get("batch_status", "Processed")
            except Exception as exc:
                result = {
                    "transaction_id": payload.get("transaction_id"),
                    "provider_id": payload.get("provider_id"),
                    "anomaly_score": None,
                    "alert_zone": "Processing Failed",
                    "is_normal": "Error",
                    "reason": f"Batch API row failed: {exc}",
                    "batch_status": "Failed",
                }
            return build_batch_result_row(index, row_dict, payload, result)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(score_one, row_info) for row_info in valid_rows]
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                total_completed = len(results)
                if total_rows:
                    progress_bar.progress(min(total_completed / total_rows, 1.0))
                status.caption(f"Processed {completed} of {len(valid_rows)} validated claims")

    progress_bar.empty()
    status.empty()

    results_df = pd.DataFrame(results)
    if not results_df.empty and "_row_order" in results_df.columns:
        results_df = results_df.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
    failures = int((results_df.get("batch_status", pd.Series(dtype=str)) == "Failed").sum())
    return results_df, failures, worker_count


def save_batch_results(results_df: pd.DataFrame) -> None:
    BATCH_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(BATCH_OUTPUT_PATH, index=False)


def render_batch_results(results_df: pd.DataFrame, failures: int, worker_count: int) -> None:
    if results_df.empty:
        st.warning("No rows were processed successfully from the uploaded CSV.")
        return

    total_rows = len(results_df)
    status_series = (
        results_df["batch_status"]
        if "batch_status" in results_df.columns
        else pd.Series(["Processed"] * len(results_df), index=results_df.index)
    )
    processed_df = results_df[status_series != "Failed"].copy()
    high_risk_count = int((results_df["alert_zone"] == "High Risk").sum())
    suspicious_count = int((results_df["alert_zone"] == "Suspicious").sum())
    flag_rate = float((processed_df["is_normal"] == "No").mean() * 100) if not processed_df.empty else 0.0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Claims", total_rows)
    metric_cols[1].metric("High Risk", high_risk_count)
    metric_cols[2].metric("Suspicious", suspicious_count)
    metric_cols[3].metric("Flag Rate", f"{flag_rate:.1f}%")

    if failures:
        st.warning(f"{failures} row(s) could not be processed. They are kept in the table and download with the error reason.")

    with st.container(border=True):
        st.markdown("#### Audit Results")
        st.caption(
            f"Compact on-screen view. Full detail remains available in the downloaded CSV. "
            f"Processed with {worker_count} parallel API request(s). Failed rows are kept with their error reason."
        )

        display_df = results_df.copy()
        display_df["review_status"] = display_df["alert_zone"].map(
            {
                "Normal": "Auto-clear",
                "Suspicious": "Review",
                "High Risk": "Escalate",
                "Processing Failed": "Check row",
            }
        ).fillna("Review")
        display_df["reason_summary"] = display_df["reason"].fillna("").map(
            lambda text: shorten(str(text), width=90, placeholder="...")
        )

        visible_columns = [
            column
            for column in [
                "transaction_id",
                "true_label",
                "predicted_label",
                "provider_id",
                "claim_type",
                "alert_zone",
                "anomaly_score",
                "review_status",
                "reason_summary",
            ]
            if column in display_df.columns
        ]

        st.dataframe(
            display_df[visible_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "transaction_id": st.column_config.TextColumn("Transaction ID", width="medium"),
                "true_label": st.column_config.TextColumn("True Label", width="small"),
                "predicted_label": st.column_config.TextColumn("Predicted Label", width="small"),
                "provider_id": st.column_config.TextColumn("Provider ID", width="small"),
                "claim_type": st.column_config.TextColumn("Claim Type", width="small"),
                "alert_zone": st.column_config.TextColumn("Zone", width="small"),
                "anomaly_score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                "review_status": st.column_config.TextColumn("Action", width="small"),
                "reason_summary": st.column_config.TextColumn("Summary", width="large"),
            },
        )

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Audit Report",
            data=csv_bytes,
            file_name="batch_inference_results.csv",
            mime="text/csv",
        )


def run_batch_audit_mode() -> None:
    st.markdown('<div class="page-section-title">Batch Audit Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-section-copy">Upload one CSV of claims and run the same live API inference used by the single-claim screen.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="page-section-title" style="font-size: 1.2rem;">Upload Claim File</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-section-copy">Drop one CSV file into the upload box or browse from your system.</div>',
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        uploaded_file = st.file_uploader(
            "Drop claim CSV here",
            type=["csv"],
            help="CSV only. One file per run. For the smoothest demo, keep it around 50 to 200 rows.",
            label_visibility="collapsed",
        )
    uploaded_signature = get_uploaded_signature(uploaded_file)

    if uploaded_file is not None:
        preview_df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        st.caption(f"Loaded {len(preview_df)} row(s) from `{uploaded_file.name}`.")
        st.caption("Preview is scrollable so you can inspect the full uploaded CSV before running the audit.")
        st.dataframe(preview_df, width="stretch", hide_index=True, height=360)

        has_cached_results = (
            uploaded_signature
            and st.session_state["batch_signature"] == uploaded_signature
            and st.session_state["batch_results"] is not None
        )

        if has_cached_results:
            st.info("Audit results for this uploaded CSV are already cached below. Upload a different CSV to run a new audit.")

        run_disabled = bool(st.session_state["batch_running"]) or bool(has_cached_results)
        if st.button("Run Batch Audit", type="primary", disabled=run_disabled):
            st.session_state["batch_running"] = True
            try:
                with st.spinner("Submitting claims to the anomaly detection API..."):
                    results_df, failures, worker_count = run_batch_inference(preview_df)
                save_batch_results(results_df)
                st.session_state["batch_results"] = results_df
                st.session_state["batch_signature"] = uploaded_signature
                st.session_state["batch_error"] = None
                st.session_state["batch_failures"] = failures
                st.session_state["batch_workers"] = worker_count
            except Exception as exc:
                st.session_state["batch_results"] = None
                st.session_state["batch_signature"] = None
                st.session_state["batch_failures"] = 0
                st.session_state["batch_workers"] = BATCH_MAX_WORKERS
                st.session_state["batch_error"] = f"Batch processing failed: {exc}"
            finally:
                st.session_state["batch_running"] = False

        if st.session_state["batch_running"]:
            st.info("Batch audit is already running. Wait for the current run to finish before submitting again.")

    if st.session_state["batch_error"]:
        st.error(st.session_state["batch_error"])

    if uploaded_signature and st.session_state["batch_signature"] == uploaded_signature and st.session_state["batch_results"] is not None:
        render_batch_results(
            st.session_state["batch_results"],
            st.session_state["batch_failures"],
            st.session_state["batch_workers"],
        )
    elif st.session_state["batch_results"] is not None and st.session_state["batch_signature"]:
        st.info("Upload the same CSV again if you want to view the cached batch results for that file.")


def main() -> None:
    init_state()
    inject_styles()
    mode = render_sidebar()
    render_header()

    if mode == "Single Claim Analysis":
        run_single_claim_form()
    else:
        run_batch_audit_mode()


if __name__ == "__main__":
    main()
