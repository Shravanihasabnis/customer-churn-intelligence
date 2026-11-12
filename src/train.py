"""End-to-end training + scoring pipeline.

Evaluation design (no leakage, temporal holdout):
  train snapshot : as_of = END - 180d, label = no order in the next 90d
  test  snapshot : as_of = END -  90d, label = no order in the next 90d (later in time)
Model selection uses 5-fold CV on the train snapshot only; the test snapshot is
reported once. The final model is refit on both snapshots and used to score
every customer as of END (the live dashboard view).

    python -m src.train
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config, features as F
from .recommend import recommend, risk_band

SEED = config.RANDOM_STATE


def make_models():
    return {
        "Logistic Regression (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    }


def snapshot(customers, tx, as_of, labelled=True):
    f = F.build_features(customers, tx, as_of)
    X = F.model_matrix(f)
    y = F.churn_labels(tx, f.index, as_of) if labelled else None
    return f, X, y


def metrics_at(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "f1": round(f1_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "brier": round(brier_score_loss(y, p), 4),
    }


def lift_table(y, p, bins=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), bins, labels=range(1, bins + 1)).astype(int)
    t = d.groupby("decile").agg(customers=("y", "size"), churn_rate=("y", "mean"), avg_pred=("p", "mean")).reset_index()
    t["lift"] = t.churn_rate / d.y.mean()
    return t.round(4)


def shap_matrix(model, X):
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer shap for RF: (n, features, classes)
        sv = sv[:, :, 1]
    base = np.ravel(ex.expected_value)[-1]
    return sv, float(base)


def main():
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(config.CUSTOMERS_CSV, parse_dates=["signup_date"])
    tx = pd.read_csv(config.TRANSACTIONS_CSV, parse_dates=["order_date"])
    end = pd.Timestamp(config.DATA_END)

    f_tr, X_tr, y_tr = snapshot(customers, tx, end - pd.Timedelta(days=180))
    f_te, X_te, y_te = snapshot(customers, tx, end - pd.Timedelta(days=90))
    print(f"train snapshot: {len(X_tr)} customers, churn {y_tr.mean():.1%} | "
          f"test snapshot: {len(X_te)} customers, churn {y_te.mean():.1%}")

    # ---- model comparison ---------------------------------------------------
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    report = {}
    for name, model in make_models().items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        report[name] = {"cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
                        "cv_roc_auc_std": round(float(cv_auc.std()), 4),
                        "test": metrics_at(y_te, p)}
        print(f"{name:32s} CV AUC {cv_auc.mean():.3f}±{cv_auc.std():.3f} | test {report[name]['test']}")

    # choose among the tree models (explainable via SHAP) using CV only
    tree_names = ["Random Forest", "XGBoost"]
    best_name = max(tree_names, key=lambda n: report[n]["cv_roc_auc_mean"])
    print("selected model (by CV):", best_name)

    # test-set diagnostics for the selected model (trained on train snapshot only)
    sel = make_models()[best_name].fit(X_tr, y_tr)
    p_te = sel.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p_te)
    keep = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
    lift = lift_table(y_te, p_te)

    # ---- final model: refit on both snapshots, score customers as of END ---
    X_all = pd.concat([X_tr, X_te], ignore_index=True)
    y_all = pd.concat([y_tr, y_te], ignore_index=True)
    final = make_models()[best_name].fit(X_all, y_all)

    f_now, X_now, _ = snapshot(customers, tx, end, labelled=False)
    p_now = final.predict_proba(X_now)[:, 1]
    sv, base = shap_matrix(final, X_now)
    shap_df = pd.DataFrame(sv, index=X_now.index, columns=X_now.columns)

    rfm = F.rfm_scores(f_now)
    cluster, cluster_profile = F.kmeans_segments(f_now)

    scored = f_now.join(rfm).join(cluster)
    scored["churn_prob"] = p_now
    scored["risk_band"] = [risk_band(p, r) for p, r in zip(p_now, scored.recency_days)]
    # revenue we can still save: only customers who are not already lost
    scored["expected_revenue_at_risk"] = np.where(
        scored.risk_band == "Lost", 0.0,
        scored.churn_prob * scored.spend_last_90d.clip(lower=scored.avg_order_value)).round(2)

    top_up, recs = [], []
    for cid, row in scored.iterrows():
        s = shap_df.loc[cid]
        drivers_up = [d for d in s.sort_values(ascending=False).index if s[d] > 0][:5]
        top_up.append(json.dumps(drivers_up))
        recs.append(json.dumps(recommend(row.to_dict(), drivers_up)))
    scored["top_risk_drivers"] = top_up
    scored["recommendations"] = recs

    # ---- persist ----------------------------------------------------------
    joblib.dump({"model": final, "features": list(X_now.columns), "name": best_name,
                 "shap_base": base}, config.ARTIFACT_DIR / "model.joblib")
    scored.reset_index(names="customer_id").to_csv(config.ARTIFACT_DIR / "scored_customers.csv", index=False)
    shap_df.reset_index(names="customer_id").to_csv(config.ARTIFACT_DIR / "shap_values.csv", index=False)
    cluster_profile.reset_index().to_csv(config.ARTIFACT_DIR / "cluster_profile.csv", index=False)
    lift.to_csv(config.ARTIFACT_DIR / "lift_table.csv", index=False)

    summary = {
        "as_of": str(end.date()), "churn_window_days": config.CHURN_WINDOW_DAYS,
        "selected_model": best_name, "n_customers_scored": int(len(scored)),
        "train_churn_rate": round(float(y_tr.mean()), 4), "test_churn_rate": round(float(y_te.mean()), 4),
        "models": report,
        "roc_curve": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        "global_importance": shap_df.abs().mean().sort_values(ascending=False).round(5).to_dict(),
        "top_decile_lift_test": float(lift.lift.iloc[0]),
    }
    (config.ARTIFACT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"scored {len(scored)} customers | risk bands: {scored.risk_band.value_counts().to_dict()}")
    print("top drivers:", list(summary["global_importance"])[:6])
    print("top-decile lift (test):", round(summary["top_decile_lift_test"], 2))


if __name__ == "__main__":
    main()
