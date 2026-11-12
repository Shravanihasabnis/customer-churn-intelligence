"""Feature engineering, RFM scoring and K-Means segmentation.

`build_features(customers, tx, as_of)` only looks at orders on/before `as_of`
so the same function is used for training snapshots and for live scoring,
which prevents target leakage.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config

NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "tenure_days",
    "avg_gap_days", "days_since_expected", "discount_ratio", "return_rate",
    "n_categories", "orders_last_30d", "orders_last_90d", "spend_last_90d",
    "order_trend", "support_tickets", "email_open_rate", "city_tier", "age",
]
CATEGORICAL_FEATURES = ["channel"]
CHANNELS = ["Organic", "Paid Ads", "Email", "Referral", "Social"]


def build_features(customers: pd.DataFrame, tx: pd.DataFrame, as_of) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    h = tx[tx.order_date <= as_of].copy()
    c = customers[customers.signup_date <= as_of]

    g = h.groupby("customer_id")
    f = pd.DataFrame({
        "last_order": g.order_date.max(),
        "first_order": g.order_date.min(),
        "frequency": g.size(),
        "monetary": g.amount.sum(),
        "discount_ratio": g.used_discount.mean(),
        "return_rate": g.returned.mean(),
        "n_categories": g.category.nunique(),
    })
    f["recency_days"] = (as_of - f.last_order).dt.days
    f["avg_order_value"] = f.monetary / f.frequency
    f["avg_gap_days"] = np.where(
        f.frequency > 1, (f.last_order - f.first_order).dt.days / (f.frequency - 1).clip(lower=1), np.nan)
    # how overdue is the customer relative to their own purchase rhythm
    f["days_since_expected"] = f.recency_days - f.avg_gap_days.fillna(f.recency_days.median())

    def window(days, col=None):
        w = h[h.order_date > as_of - pd.Timedelta(days=days)].groupby("customer_id")
        return (w.size() if col is None else w[col].sum()).reindex(f.index).fillna(0)

    f["orders_last_30d"] = window(30)
    f["orders_last_90d"] = window(90)
    f["spend_last_90d"] = window(90, "amount")
    prev = h[(h.order_date <= as_of - pd.Timedelta(days=90)) &
             (h.order_date > as_of - pd.Timedelta(days=180))].groupby("customer_id").size()
    prev = prev.reindex(f.index).fillna(0)
    f["order_trend"] = (f.orders_last_90d - prev) / (prev + 1)   # >0 growing, <0 declining

    f = f.join(c.set_index("customer_id")[["signup_date", "channel", "city_tier", "age",
                                            "support_tickets", "email_open_rate"]], how="inner")
    f["tenure_days"] = (as_of - f.signup_date).dt.days
    f["avg_gap_days"] = f.avg_gap_days.fillna(f.avg_gap_days.median())
    return f.drop(columns=["last_order", "first_order", "signup_date"])


def churn_labels(tx: pd.DataFrame, customer_ids, as_of, window_days=config.CHURN_WINDOW_DAYS) -> pd.Series:
    """1 = no purchase in (as_of, as_of + window]."""
    as_of = pd.Timestamp(as_of)
    fut = tx[(tx.order_date > as_of) & (tx.order_date <= as_of + pd.Timedelta(days=window_days))]
    active = set(fut.customer_id)
    return pd.Series([0 if cid in active else 1 for cid in customer_ids],
                     index=customer_ids, name="churned")


def model_matrix(f: pd.DataFrame) -> pd.DataFrame:
    """Numeric features + one-hot channel with a fixed column set."""
    X = f[NUMERIC_FEATURES].copy()
    for ch in CHANNELS:
        X[f"channel_{ch}"] = (f.channel == ch).astype(int)
    return X.astype(float)


# ------------------------------------------------------------------ RFM ----
def rfm_scores(f: pd.DataFrame) -> pd.DataFrame:
    r = pd.DataFrame(index=f.index)
    r["R"] = pd.qcut(f.recency_days.rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    r["F"] = pd.qcut(f.frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    r["M"] = pd.qcut(f.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    r["rfm_score"] = r.R * 100 + r.F * 10 + r.M
    r["rfm_segment"] = [_rfm_segment(a, b, c) for a, b, c in zip(r.R, r.F, r.M)]
    return r


def _rfm_segment(R, F, M):
    if R >= 4 and F >= 4 and M >= 4:
        return "Champions"
    if R >= 3 and F >= 4:
        return "Loyal"
    if R >= 4 and F <= 2:
        return "New / Recent"
    if R >= 3 and F >= 2:
        return "Potential Loyalists"
    if R <= 2 and F >= 4:
        return "Can't Lose Them"
    if R <= 2 and F >= 2:
        return "At Risk"
    return "Hibernating"


# ------------------------------------------------------------- K-Means -----
def kmeans_segments(f: pd.DataFrame, k: int = config.N_CLUSTERS):
    cols = ["recency_days", "frequency", "monetary", "avg_order_value", "discount_ratio"]
    Z = StandardScaler().fit_transform(np.log1p(f[cols].clip(lower=0)))
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(Z)
    labels = km.labels_
    prof = f.assign(cluster=labels).groupby("cluster")[["recency_days", "frequency", "monetary", "discount_ratio"]].mean()
    names = _name_clusters(prof)
    return pd.Series([names[l] for l in labels], index=f.index, name="cluster_name"), prof.assign(
        customers=pd.Series(labels).value_counts().sort_index(), name=pd.Series(names))


def _name_clusters(prof: pd.DataFrame) -> dict:
    """Name clusters from their centroids; each name is used at most once."""
    remaining = set(prof.index)
    names = {}

    def pick(label, score):
        c = max(remaining, key=lambda i: score.loc[i])
        names[c] = label
        remaining.discard(c)

    pick("High-Value Regulars", (prof.monetary.rank() + prof.frequency.rank() - prof.recency_days.rank()))
    if remaining:
        pick("Slipping Regulars", prof.monetary)          # best remaining spenders, but not recent
    if remaining:
        pick("Dormant", prof.recency_days)                # longest since last order
    for c in remaining:
        names[c] = "Low-Value Casual"
    return names
