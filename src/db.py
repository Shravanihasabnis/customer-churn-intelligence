"""SQLite store for scored customers, SHAP values and transactions.

SQLite keeps the project zero-config. The API only uses SQL through
`query()`, so swapping to PostgreSQL means changing the connection here
(e.g. SQLAlchemy engine from DATABASE_URL).

    python -m src.db      # (re)build artifacts/store.db from the CSV outputs
"""
import sqlite3

import pandas as pd

from . import config

DB_PATH = config.ARTIFACT_DIR / "store.db"


def build():
    scored = pd.read_csv(config.ARTIFACT_DIR / "scored_customers.csv")
    shap_vals = pd.read_csv(config.ARTIFACT_DIR / "shap_values.csv")
    tx = pd.read_csv(config.TRANSACTIONS_CSV)
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        scored.to_sql("customers", con, index=False)
        shap_vals.to_sql("shap_values", con, index=False)
        tx.to_sql("transactions", con, index=False)
        con.execute("CREATE INDEX idx_c_id ON customers(customer_id)")
        con.execute("CREATE INDEX idx_s_id ON shap_values(customer_id)")
        con.execute("CREATE INDEX idx_t_cust ON transactions(customer_id)")
    print(f"built {DB_PATH} ({len(scored)} customers, {len(tx)} transactions)")


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    if not DB_PATH.exists():
        build()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(sql, con, params=params)


if __name__ == "__main__":
    build()
