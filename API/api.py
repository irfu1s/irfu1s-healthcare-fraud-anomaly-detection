from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model_inference import run_inference

app = FastAPI(title="Healthcare Anomaly Detection API")

# ---------- INPUT SCHEMA ----------
class Transaction(BaseModel):
    # 1. Core Identifiers & Financials
    transaction_id: int
    provider_id: str
    claim_type: str
    billing_amount: float
    deductible_context: float
    coinsurance_context: float
    
    # 2. Dates & Demographics
    service_start: str
    service_end: str
    patient_age: int
    
    # 3. Medical Context
    diagnosis_context: str
    has_diabetes: int
    has_chf: int
    has_cancer: int
    has_copd: int
    primary_procedure: str
    treatment_group: str
    
    # 4. Invisible Background Noise
    patient_id: str
    admission_date: str
    gender_context: int
    demographic_context: int
    primary_service: str


class BatchRequest(BaseModel):
    transactions: list[Transaction]


# ---------- HEALTH CHECK ----------
@app.get("/")
def home():
    return {"status": "API is online. Ready to hunt fraud."}

# ---------- INFERENCE ENDPOINT ----------
@app.post("/evaluate_transaction")
def predict(txn: Transaction, include_explainability: bool = True):
    try:
        new_transaction = txn.model_dump()
        
        print(f"--> Received {new_transaction['claim_type']} Claim: Provider {new_transaction['provider_id']} | Amount: ${new_transaction['billing_amount']}")
        
        result = run_inference(new_transaction, include_explainability=include_explainability)
        
        return result
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")


@app.post("/evaluate_batch")
def predict_batch(batch: BatchRequest):
    results = []

    for index, txn in enumerate(batch.transactions, start=1):
        new_transaction = txn.model_dump()
        try:
            if index == 1 or index % 10 == 0 or index == len(batch.transactions):
                print(
                    f"--> Batch {index}/{len(batch.transactions)}: "
                    f"{new_transaction['claim_type']} Claim | Provider {new_transaction['provider_id']} | "
                    f"Amount: ${new_transaction['billing_amount']}"
                )
            result = run_inference(new_transaction, include_explainability=False)
            result["batch_status"] = "Processed"
            results.append(result)
        except Exception as e:
            print(f"Batch API Error at row {index}: {str(e)}")
            results.append(
                {
                    "transaction_id": new_transaction.get("transaction_id"),
                    "provider_id": new_transaction.get("provider_id"),
                    "anomaly_score": None,
                    "alert_zone": "Processing Failed",
                    "is_normal": "Error",
                    "reason": f"Inference Engine Error: {str(e)}",
                    "batch_status": "Failed",
                }
            )

    failed_count = sum(1 for result in results if result.get("batch_status") == "Failed")
    return {
        "results": results,
        "processed_count": len(results) - failed_count,
        "failed_count": failed_count,
        "total_count": len(results),
    }
