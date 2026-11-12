"""Business analytics computed from raw transactions (cohort retention)."""
import pandas as pd


def cohort_retention(tx: pd.DataFrame, months: int = 6, last_cohorts: int = 8) -> pd.DataFrame:
    """Monthly cohort retention: share of a cohort (first-order month) ordering again in month k.

    Returns rows = cohort month, columns = 0..months (M0 is always 100%).
    Cells beyond the last observed month are left NaN (not 0%).
    """
    tx = tx.assign(order_date=pd.to_datetime(tx.order_date))
    tx["month"] = tx.order_date.dt.to_period("M")
    first = tx.groupby("customer_id").month.min().rename("cohort")
    tx = tx.join(first, on="customer_id")
    tx["k"] = (tx.month - tx.cohort).apply(lambda x: x.n)
    last_month = tx.month.max()

    active = tx.groupby(["cohort", "k"]).customer_id.nunique().unstack(fill_value=0)
    size = active[0]
    ret = active.div(size, axis=0)
    ret = ret.reindex(columns=range(months + 1))
    for cohort in ret.index:                      # mask months that haven't happened yet
        for k in ret.columns:
            if cohort + k > last_month:
                ret.loc[cohort, k] = float("nan")
    ret = ret.tail(last_cohorts)
    ret.index = ret.index.astype(str)
    ret.insert(0, "cohort_size", size.reindex(ret.index.map(pd.Period)).values)
    return ret.round(4)
