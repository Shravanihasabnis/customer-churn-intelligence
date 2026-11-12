import pandas as pd
import pytest

from src import config, features as F
from src.recommend import recommend, risk_band


def _toy():
    customers = pd.DataFrame({
        "customer_id": ["a", "b"], "signup_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "channel": ["Organic", "Email"], "city_tier": [1, 2], "age": [30, 40],
        "support_tickets": [0, 3], "email_open_rate": [0.4, 0.1]})
    tx = pd.DataFrame({
        "order_id": list("12345"), "customer_id": ["a", "a", "a", "b", "b"],
        "order_date": pd.to_datetime(["2025-02-01", "2025-03-01", "2025-06-15", "2025-02-10", "2025-08-01"]),
        "category": ["Home"] * 5, "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        "used_discount": [0, 1, 0, 0, 0], "returned": [0, 0, 0, 1, 0]})
    return customers, tx


def test_features_ignore_future_orders():
    c, tx = _toy()
    f = F.build_features(c, tx, "2025-04-01")
    assert f.loc["a", "frequency"] == 2          # June order must not leak in
    assert "b" in f.index and f.loc["b", "frequency"] == 1
    assert f.loc["a", "recency_days"] == 31


def test_churn_label_uses_only_next_window():
    c, tx = _toy()
    y = F.churn_labels(tx, ["a", "b"], "2025-04-01", window_days=90)   # (Apr 1, Jun 30]
    assert y["a"] == 0                            # ordered Jun 15
    assert y["b"] == 1                            # next order is Aug 1, outside window


def test_model_matrix_has_fixed_columns():
    c, tx = _toy()
    X = F.model_matrix(F.build_features(c, tx, "2025-09-01"))
    assert X.shape[1] == len(F.NUMERIC_FEATURES) + len(F.CHANNELS)
    assert not X.isna().any().any()


def test_risk_bands():
    assert risk_band(0.9, 10) == "High"
    assert risk_band(0.5, 10) == "Medium"
    assert risk_band(0.1, 10) == "Low"
    assert risk_band(0.99, config.LOST_AFTER_DAYS + 1) == "Lost"


def test_recommendations():
    row = {"churn_prob": 0.9, "rfm_segment": "At Risk", "monetary": 900, "recency_days": 60}
    acts = recommend(row, ["support_tickets", "recency_days"])
    assert acts[0]["priority"] == "Critical"      # high value + high risk is flagged first
    assert any(a["action"] == "Service recovery" for a in acts)
    lost = recommend({**row, "recency_days": 400}, [])
    assert "eactivation" in lost[0]["action"]


# ---- API (skipped if artifacts haven't been built) -------------------------
@pytest.fixture(scope="module")
def client():
    if not (config.ARTIFACT_DIR / "model.joblib").exists():
        pytest.skip("run `python -m src.train` first")
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_api_basic(client):
    assert client.get("/health").json()["status"] == "ok"
    s = client.get("/summary").json()
    assert s["customers"] == s["high_risk"] + s["medium_risk"] + s["low_risk"] + s["lost"]
    items = client.get("/customers", params={"risk": "High", "limit": 3}).json()["items"]
    assert items and all(i["risk_band"] == "High" for i in items)
    cid = items[0]["customer_id"]
    assert client.get(f"/customers/{cid}").json()["recommendations"]
    ex = client.get(f"/customers/{cid}/explanation").json()
    assert len(ex["contributions"]) == 8
    assert client.get("/customers/NOPE").status_code == 404


def test_api_predict_monotonic(client):
    base = dict(frequency=10, monetary=500, orders_last_90d=2, orders_last_30d=1)
    fresh = client.post("/predict", json={**base, "recency_days": 5}).json()["churn_probability"]
    stale = client.post("/predict", json={**base, "recency_days": 150}).json()["churn_probability"]
    assert stale > fresh
    assert client.post("/predict", json={**base, "recency_days": 5, "channel": "Fax"}).status_code == 422
