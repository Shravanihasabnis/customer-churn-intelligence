"""Central configuration: paths and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"

# Time-based labelling: features use history up to CUTOFF, the churn label is
# "no purchase in the following CHURN_WINDOW_DAYS days". This avoids leakage.
CHURN_WINDOW_DAYS = 90
DATA_START = "2024-01-01"
DATA_END = "2025-12-31"

N_CUSTOMERS = 6000
N_CLUSTERS = 4
RANDOM_STATE = 42

# Risk bands on churn probability
RISK_HIGH = 0.65
RISK_MEDIUM = 0.35

# Customers with no order for this long are treated as already lost (reactivation, not retention)
LOST_AFTER_DAYS = 180
