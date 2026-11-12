"""FastAPI backend.

    uvicorn api.main:app --reload
    open http://localhost:8000/docs
"""
import json
from functools import lru_cache
from typing import Optional

try:
    import joblib
except ImportError:  # pragma: no cover - optional at static-analysis time
    joblib = None

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import analytics, config, db, features as F
from src.recommend import recommend, risk_band

app = FastAPI(title="E-Commerce Customer Intelligence & Retention API", version="1.0.0")


# ------------------------------------------------------------- resources ----
@lru_cache(maxsize=1)
def bundle():
    return joblib.load(config.ARTIFACT_DIR / "model.joblib")


@lru_cache(maxsize=1)
def explainer():
    return shap.TreeExplainer(bundle()["model"])


@lru_cache(maxsize=1)
def metrics():
    return json.loads((config.ARTIFACT_DIR / "metrics.json").read_text())


def _decode(df: pd.DataFrame) -> list[dict]:
    out = df.replace({np.nan: None}).to_dict(orient="records")
    for r in out:
        for k in ("top_risk_drivers", "recommendations"):
            if isinstance(r.get(k), str):
                r[k] = json.loads(r[k])
    return out


# ---------------------------------------------------------------- routes ----
@app.get("/health")
def health():
    return {"status": "ok", "model": bundle()["name"], "as_of": metrics()["as_of"]}


@app.get("/summary")
def summary():
    d = db.query("SELECT churn_prob, risk_band, monetary, expected_revenue_at_risk FROM customers")
    return {
        "customers": int(len(d)),
        "high_risk": int((d.risk_band == "High").sum()),
        "medium_risk": int((d.risk_band == "Medium").sum()),
        "low_risk": int((d.risk_band == "Low").sum()),
        "lost": int((d.risk_band == "Lost").sum()),
        "avg_churn_probability_active": round(float(d[d.risk_band != "Lost"].churn_prob.mean()), 4),
        "expected_revenue_at_risk": round(float(d.expected_revenue_at_risk.sum()), 2),
        "total_lifetime_revenue": round(float(d.monetary.sum()), 2),
        "as_of": metrics()["as_of"],
        "churn_window_days": config.CHURN_WINDOW_DAYS,
    }


@app.get("/metrics")
def model_metrics():
    return metrics()


@app.get("/segments")
def segments():
    rfm = db.query(
        "SELECT rfm_segment AS segment, COUNT(*) AS customers, ROUND(AVG(churn_prob),4) AS avg_churn_prob, "
        "ROUND(SUM(monetary),2) AS revenue, ROUND(AVG(monetary),2) AS avg_clv, ROUND(AVG(frequency),2) AS avg_orders, "
        "ROUND(AVG(avg_order_value),2) AS aov FROM customers GROUP BY rfm_segment ORDER BY customers DESC")
    km = db.query(
        "SELECT cluster_name AS cluster, COUNT(*) AS customers, ROUND(AVG(churn_prob),4) AS avg_churn_prob, "
        "ROUND(AVG(recency_days),1) AS avg_recency_days, ROUND(AVG(frequency),1) AS avg_orders, "
        "ROUND(AVG(monetary),2) AS avg_spend FROM customers GROUP BY cluster_name ORDER BY customers DESC")
    return {"rfm": rfm.to_dict("records"), "kmeans": km.to_dict("records")}


@app.get("/cohorts")
def cohorts(months: int = Query(6, ge=1, le=12), last: int = Query(8, ge=1, le=24)):
    tx = db.query("SELECT customer_id, order_date FROM transactions")
    r = analytics.cohort_retention(tx, months=months, last_cohorts=last)
    return {"cohorts": r.reset_index().replace({np.nan: None}).to_dict("records")}


@app.get("/customers")
def customers(risk: Optional[str] = Query(None, pattern="^(High|Medium|Low|Lost)$"),
              segment: Optional[str] = None,
              sort: str = Query("churn_prob", pattern="^(churn_prob|monetary|expected_revenue_at_risk|recency_days|frequency)$"),
              limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    where, params = [], []
    if risk:
        where.append("risk_band = ?"); params.append(risk)
    if segment:
        where.append("rfm_segment = ?"); params.append(segment)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    total = int(db.query(f"SELECT COUNT(*) n FROM customers {clause}", tuple(params)).n[0])
    cols = ("customer_id, frequency AS orders, ROUND(monetary,2) AS ltv, recency_days, rfm_segment, cluster_name, "
            "ROUND(churn_prob,4) AS churn_prob, risk_band, expected_revenue_at_risk")
    d = db.query(f"SELECT {cols} FROM customers {clause} ORDER BY {sort} DESC LIMIT ? OFFSET ?",
                 tuple(params) + (limit, offset))
    return {"total": total, "items": _decode(d)}


@app.get("/customers/{customer_id}")
def customer(customer_id: str):
    d = db.query("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    if d.empty:
        raise HTTPException(404, f"customer {customer_id} not found")
    return _decode(d)[0]


@app.get("/customers/{customer_id}/explanation")
def explanation(customer_id: str, top: int = Query(8, ge=1, le=30)):
    s = db.query("SELECT * FROM shap_values WHERE customer_id = ?", (customer_id,))
    c = db.query("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    if s.empty or c.empty:
        raise HTTPException(404, f"customer {customer_id} not found")
    sv = s.drop(columns="customer_id").iloc[0]
    X = F.model_matrix(c.assign(channel=c.channel)).iloc[0]
    order = sv.abs().sort_values(ascending=False).index[:top]
    base = bundle()["shap_base"]
    return {
        "customer_id": customer_id,
        "churn_prob": round(float(c.churn_prob[0]), 4),
        "base_value": round(base, 4),
        "contributions": [{"feature": f, "value": round(float(X[f]), 3),
                           "shap": round(float(sv[f]), 4)} for f in order],
        "other_features_shap": round(float(sv.drop(order).sum()), 4),
        "note": "SHAP values are in probability units for Random Forest (log-odds for XGBoost).",
    }


class PredictRequest(BaseModel):
    """Raw customer features as of today. Missing optional fields fall back to sensible defaults."""
    recency_days: float = Field(..., ge=0)
    frequency: float = Field(..., ge=1)
    monetary: float = Field(..., ge=0)
    tenure_days: float = Field(365, ge=0)
    avg_gap_days: float = Field(30, ge=0)
    discount_ratio: float = Field(0.3, ge=0, le=1)
    return_rate: float = Field(0.08, ge=0, le=1)
    n_categories: float = Field(2, ge=1)
    orders_last_30d: float = Field(0, ge=0)
    orders_last_90d: float = Field(1, ge=0)
    spend_last_90d: float = Field(0, ge=0)
    order_trend: float = 0.0
    support_tickets: float = Field(0, ge=0)
    email_open_rate: float = Field(0.3, ge=0, le=1)
    city_tier: int = Field(2, ge=1, le=3)
    age: int = Field(33, ge=18, le=100)
    channel: str = "Organic"


@app.post("/predict")
def predict(req: PredictRequest):
    """Score a hypothetical or new customer and return drivers + recommended actions."""
    r = req.model_dump()
    if r["channel"] not in F.CHANNELS:
        raise HTTPException(422, f"channel must be one of {F.CHANNELS}")
    r["avg_order_value"] = r["monetary"] / r["frequency"]
    r["days_since_expected"] = r["recency_days"] - r["avg_gap_days"]
    f = pd.DataFrame([r])
    X = F.model_matrix(f)[bundle()["features"]]
    p = float(bundle()["model"].predict_proba(X)[:, 1][0])
    sv = np.asarray(explainer().shap_values(X))
    sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    contrib = pd.Series(sv, index=X.columns)
    drivers_up = [d for d in contrib.sort_values(ascending=False).index if contrib[d] > 0][:5]
    # RFM segment for a single customer needs the population: use quintile cutoffs from stored data
    pop = db.query("SELECT recency_days, frequency, monetary FROM customers")
    R = 5 - int(np.searchsorted(np.quantile(pop.recency_days, [.2, .4, .6, .8]), r["recency_days"]))
    Fq = 1 + int(np.searchsorted(np.quantile(pop.frequency, [.2, .4, .6, .8]), r["frequency"]))
    Mq = 1 + int(np.searchsorted(np.quantile(pop.monetary, [.2, .4, .6, .8]), r["monetary"]))
    R, Fq, Mq = (min(max(v, 1), 5) for v in (R, Fq, Mq))
    seg = F._rfm_segment(R, Fq, Mq)
    row = {"churn_prob": p, "rfm_segment": seg, "monetary": r["monetary"], "recency_days": r["recency_days"]}
    return {
        "churn_probability": round(p, 4), "risk_band": risk_band(p, r["recency_days"]), "rfm_segment": seg,
        "top_risk_drivers": drivers_up,
        "contributions": {k: round(float(v), 4) for k, v in contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8]).items()},
        "recommendations": recommend(row, drivers_up),
    }
