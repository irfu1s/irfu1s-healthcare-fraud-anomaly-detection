import sqlite3
import pandas as pd

DB_PATH = ("database/healthcare.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

# -------- TRAINING USE --------
def load_full_dataset():
    # Using 'with' automatically safely closes the connection when done
    with get_connection() as conn:
        # parse_dates forces Pandas to build the time-series math your LSTM needs immediately
        df = pd.read_sql(
            "SELECT * FROM Healthcare_transactions", 
            conn, 
            parse_dates=['service_start', 'service_end', 'admission_date', 'patient_dob']
        )
    return df

# -------- INFERENCE USE --------
def load_provider_history(provider_id):
    with get_connection() as conn:
        df = pd.read_sql(
            """
            SELECT * FROM Healthcare_transactions
            WHERE provider_id = ?
            ORDER BY service_start
            """,
            conn,
            params=(provider_id,),
            parse_dates=['service_start', 'service_end', 'admission_date', 'patient_dob']
        )
    return df

# -------- WRITE BACK --------
def write_scores(df):
    with get_connection() as conn:
        # Ready to catch your IF/LSTM anomaly flags and XAI explanations
        df.to_sql("scored_transactions", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
