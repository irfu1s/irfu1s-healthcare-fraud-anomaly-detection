import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
db_filename = BASE_DIR / "database" / "healthcare.db"

print(f"Connecting to {db_filename}...")
conn = sqlite3.connect(db_filename)

query = "SELECT * FROM Healthcare_transactions"

print("Loading data into Pandas. This might take a few seconds...")
df = pd.read_sql_query(query, conn)

conn.close()

print("\n--- DATASET SUCCESSFULLY LOADED ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n--- FIRST 3 ROWS ---")
print(df.head(3))

print("\n--- DATA TYPES ---")
print(df.dtypes)
