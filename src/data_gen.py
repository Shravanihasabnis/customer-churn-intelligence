"""Synthetic e-commerce data generator.

Each customer has latent traits (engagement, price sensitivity, signup date).
Purchases follow a Poisson process whose rate decays as the customer becomes
disengaged; some customers permanently stop buying (true churn). Support
tickets and returns raise churn hazard, discounts attract price-sensitive users.
The signal is noisy on purpose so model metrics are realistic, not perfect.
"""
import numpy as np
import pandas as pd

from . import config

CATEGORIES = ["Electronics", "Fashion", "Home", "Beauty", "Grocery", "Sports"]
CHANNELS = ["Organic", "Paid Ads", "Email", "Referral", "Social"]


def generate(n_customers: int = config.N_CUSTOMERS, seed: int = config.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(config.DATA_START)
    end = pd.Timestamp(config.DATA_END)
    span = (end - start).days

    # ---- customers -------------------------------------------------------
    # Signups skew earlier so there is a mix of tenures.
    signup_offset = (rng.beta(1.2, 1.8, n_customers) * (span - 20)).astype(int)
    signup = start + pd.to_timedelta(signup_offset, unit="D")
    customers = pd.DataFrame({
        "customer_id": [f"C-{10000 + i}" for i in range(n_customers)],
        "signup_date": signup,
        "channel": rng.choice(CHANNELS, n_customers, p=[0.3, 0.25, 0.15, 0.15, 0.15]),
        "city_tier": rng.choice([1, 2, 3], n_customers, p=[0.4, 0.35, 0.25]),
        "age": np.clip(rng.normal(33, 9, n_customers), 18, 70).astype(int),
    })
    engagement = rng.gamma(2.0, 0.5, n_customers)              # latent, mean ~1
    price_sens = rng.beta(2, 3, n_customers)                    # 0..1
    fav_cat = rng.choice(CATEGORIES, n_customers)
    # day (since signup) after which the customer silently churns; inf = never
    churn_hazard = np.clip(0.0009 / np.maximum(engagement, 0.2) + 0.0006 * price_sens, 0.0002, 0.01)
    churn_after = rng.exponential(1.0 / churn_hazard)

    # ---- transactions ----------------------------------------------------
    rows = []
    order_id = 0
    for i in range(n_customers):
        s = signup_offset[i]
        base_rate = 0.045 * engagement[i]                       # orders/day
        last_day = min(span, s + int(churn_after[i]))
        t = s
        while True:
            # inter-purchase gap, longer as customer approaches churn
            progress = (t - s) / max(churn_after[i], 1)
            rate = base_rate * (1 - 0.5 * min(progress, 1.0))
            t += max(1, int(rng.exponential(1.0 / max(rate, 0.002))))
            if t > last_day:
                break
            discount = rng.random() < (0.15 + 0.5 * price_sens[i])
            cat = fav_cat[i] if rng.random() < 0.6 else rng.choice(CATEGORIES)
            base_amt = {"Electronics": 90, "Fashion": 45, "Home": 55,
                        "Beauty": 28, "Grocery": 22, "Sports": 50}[cat]
            amt = rng.lognormal(np.log(base_amt), 0.45) * (0.85 if discount else 1.0)
            returned = rng.random() < (0.05 + 0.08 * price_sens[i])
            rows.append((f"O-{order_id}", customers.customer_id[i], start + pd.Timedelta(days=int(t)),
                         cat, round(float(amt), 2), int(discount), int(returned)))
            order_id += 1
    tx = pd.DataFrame(rows, columns=["order_id", "customer_id", "order_date",
                                     "category", "amount", "used_discount", "returned"])

    # Support tickets: more likely for customers about to churn (weak signal)
    near_churn = (signup_offset + churn_after) < span
    lam = 0.4 + 1.2 * near_churn + 0.4 * (1 - np.clip(engagement, 0, 2) / 2)
    customers["support_tickets"] = rng.poisson(lam)
    customers["email_open_rate"] = np.clip(
        rng.normal(0.25 + 0.15 * np.clip(engagement, 0, 2) - 0.1 * near_churn, 0.12), 0, 1).round(3)

    # keep only customers with at least one order (others can't be scored)
    tx = tx.sort_values("order_date").reset_index(drop=True)
    customers = customers[customers.customer_id.isin(tx.customer_id.unique())].reset_index(drop=True)
    return customers, tx


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    customers, tx = generate()
    customers.to_csv(config.CUSTOMERS_CSV, index=False)
    tx.to_csv(config.TRANSACTIONS_CSV, index=False)
    print(f"customers={len(customers):,} transactions={len(tx):,} "
          f"range={tx.order_date.min().date()}..{tx.order_date.max().date()}")


if __name__ == "__main__":
    main()
