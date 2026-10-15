
import sqlite3
import pandas as pd
from pathlib import Path

csv_file = Path("customer_purchase_data.csv")
db_file = Path("customer_purchases.db")

if not csv_file.exists():
    raise FileNotFoundError("customer_purchase_data.csv not found!")

df = pd.read_csv(csv_file)

with sqlite3.connect(db_file) as conn:
    df.to_sql(
        "purchases",
        conn,
        if_exists="replace",
        index=False
    )

    count = pd.read_sql_query(
        "SELECT COUNT(*) AS total FROM purchases",
        conn
    )

print("Database created successfully!")
print("Database file:", db_file)
print("Records imported:", int(count.iloc[0, 0]))
