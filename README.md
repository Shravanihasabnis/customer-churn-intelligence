# AI-Powered E-Commerce Customer Intelligence & Retention System

End-to-end project: behaviour analytics → RFM + K-Means segmentation → churn prediction → SHAP explanations →
rule-based retention actions → FastAPI backend → Streamlit dashboard → Docker.

> **Data is synthetic.** `src/data_gen.py` simulates ~5.5k customers and ~83k orders with realistic churn behaviour
> (purchase rate decay, discount sensitivity, returns, support tickets). Swap in a real dataset (e.g. UCI Online Retail II)
> by producing the same two CSVs: `customers.csv` and `transactions.csv` (see columns in `data_gen.py`).

## Architecture

```
data_gen.py ─► data/*.csv ─► features.py (features, RFM, K-Means) ─► train.py (RF / XGBoost / LR, SHAP, scoring)
                                                                          │
                                                    artifacts/ (model, scores, SHAP) ─► db.py (SQLite)
                                                                                          │
                                         recommend.py (rules) ─► api/main.py (FastAPI) ─► dashboard/app.py (Streamlit)
```

| Module | Where | Notes |
|---|---|---|
| Behaviour analysis | `src/features.py` | 18 features: recency, frequency, monetary, trend, overdue-vs-own-rhythm, discount/return rates, support tickets… |
| Segmentation | `src/features.py` | RFM quintile scores → 7 named segments; K-Means (k=4) on log-scaled RFM-type features |
| Churn prediction | `src/train.py` | Logistic baseline vs Random Forest vs XGBoost |
| Explainability | `src/train.py`, API | SHAP TreeExplainer: global importance + per-customer waterfall |
| Retention engine | `src/recommend.py` | Segment playbooks + driver-specific actions + value-at-risk priority |
| Backend | `api/main.py` | FastAPI, OpenAPI docs at `/docs` |
| Database | `src/db.py` | SQLite (swap connection for PostgreSQL) |
| Dashboard | `dashboard/app.py` | 5 pages: Customers & Cohorts, Churn Risk, Explain, Retention Actions, Model Performance |
| Deployment | `Dockerfile`, `docker-compose.yml` | API + dashboard |

## Modelling choices worth knowing (interview talking points)

- **Churn definition:** no order in the next 90 days. Features only use orders up to the snapshot date (`build_features(as_of)`), so there is no target leakage; a test enforces this.
- **Temporal validation:** train on a snapshot 180 days before the end of the data, test on the snapshot 90 days before it, not a random split. Model selection uses 5-fold CV on the training snapshot only; the test set is looked at once.
- **Honest baseline:** a logistic regression baseline is included. On this synthetic data it matches or slightly beats RF/XGBoost (all ≈0.94 ROC-AUC) because recency dominates the signal. Say so rather than claiming XGBoost "wins".
- **Lost vs at-risk:** customers with no order in 180+ days are "Lost" (reactivation), so risk bands and revenue-at-risk only cover customers who can still be saved.
- **Final model** is refit on both snapshots and scores every customer as of the last date.

## Run locally

```bash
pip install -r requirements.txt
python -m src.data_gen      # data/*.csv
python -m src.train         # artifacts/ (≈5 min: CV + SHAP)
python -m src.db            # artifacts/store.db

uvicorn api.main:app --port 8000          # http://localhost:8000/docs
streamlit run dashboard/app.py            # http://localhost:8501   (API_URL env var to point elsewhere)

pytest -q
```

## Docker

```bash
docker compose up --build     # API :8000, dashboard :8501 (image trains the model at build time)
```

## Deploy

Push to GitHub, then on Render create two Docker web services from this repo: API (default CMD, port 8000) and
dashboard (start command `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`, env `API_URL=<api url>`).

## API

`GET /summary` · `/segments` · `/cohorts` · `/customers?risk=High&segment=At%20Risk&sort=expected_revenue_at_risk` ·
`/customers/{id}` · `/customers/{id}/explanation` · `/metrics` · `POST /predict` (what-if scoring with drivers and actions)

## Limitations

- Synthetic data: metrics show the pipeline works, not real-world performance.
- Retention rules are heuristics; measuring uplift needs an A/B test, which this project doesn't simulate.
- SQLite is single-node; use PostgreSQL for multi-user deployments.
