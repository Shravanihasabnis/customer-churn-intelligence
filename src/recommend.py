"""Rule-based retention recommendation engine.

Combines churn risk, RFM segment and the top SHAP drivers of a customer into
an ordered list of actions. Rules are explicit so they can be audited and
changed by a marketing team without retraining anything.
"""
from . import config

# Human-readable meaning of model features, used when a driver is pushing risk up
DRIVER_ACTIONS = {
    "recency_days": ("Win-back nudge", "Hasn't ordered in a while: send a personalised 'we miss you' email with a time-limited offer."),
    "days_since_expected": ("Reminder at their usual cadence", "Overdue versus their own buying rhythm: trigger a replenishment / back-in-stock reminder."),
    "order_trend": ("Re-engage declining activity", "Order activity is dropping: recommend products from their favourite categories."),
    "orders_last_90d": ("Re-engage declining activity", "Few recent orders: surface bestsellers and bundles."),
    "orders_last_30d": ("Re-engage declining activity", "No orders this month: send a curated picks email."),
    "support_tickets": ("Service recovery", "Multiple support tickets: proactive outreach from support plus a goodwill credit."),
    "return_rate": ("Fix product-fit issues", "High returns: share size/fit guides and quality-assured picks."),
    "email_open_rate": ("Change channel", "Ignores emails: try SMS / push / WhatsApp instead of email."),
    "discount_ratio": ("Protect margin", "Buys mostly on discount: offer loyalty points or free shipping instead of deeper discounts."),
}

SEGMENT_PLAYBOOK = {
    "Champions": ("VIP programme", "Early access, referral rewards and a thank-you gift. Don't discount."),
    "Loyal": ("Upsell", "Cross-sell complementary categories and invite to the loyalty tier."),
    "Potential Loyalists": ("Build habit", "Offer a second-purchase incentive and subscription / bundle options."),
    "New / Recent": ("Onboard", "Welcome series and a first-repeat-purchase incentive within 30 days."),
    "At Risk": ("Win-back", "Personalised reminder with a moderate offer before they lapse completely."),
    "Can't Lose Them": ("High-touch win-back", "Personal outreach (call / VIP email) and a strong offer: these were valuable."),
    "Hibernating": ("Low-cost reactivation", "One low-cost reactivation campaign; suppress from paid retargeting if no response."),
}


def risk_band(p: float, recency_days: float = 0) -> str:
    if recency_days > config.LOST_AFTER_DAYS:
        return "Lost"
    return "High" if p >= config.RISK_HIGH else "Medium" if p >= config.RISK_MEDIUM else "Low"


def recommend(row: dict, top_drivers: list[str], max_actions: int = 3) -> list[dict]:
    """row: a customer's feature/score row. top_drivers: features pushing churn risk UP, strongest first."""
    p = row["churn_prob"]
    band = risk_band(p, row.get("recency_days", 0))
    seg = row["rfm_segment"]
    actions = []

    if band == "Lost":
        value = row.get("monetary", 0)
        if value >= 500:
            return [{"action": "Reactivation: high-value lapsed",
                     "detail": f"Lifetime spend {value:.0f} but no order in {row['recency_days']:.0f} days: personal outreach plus a strong comeback offer.",
                     "priority": "Medium"}]
        return [{"action": "Low-cost reactivation",
                 "detail": "One automated win-back email; suppress from paid retargeting if there is no response.",
                 "priority": "Low"}]

    if band == "Low":
        title, text = SEGMENT_PLAYBOOK.get(seg, ("Nurture", "Keep in regular lifecycle communications."))
        return [{"action": title, "detail": text, "priority": "Low"}]

    # 1. Segment-level playbook gets top billing for valuable customers
    title, text = SEGMENT_PLAYBOOK.get(seg, ("Win-back", "Targeted retention offer."))
    actions.append({"action": title, "detail": text, "priority": "High" if band == "High" else "Medium"})

    # 2. Driver-specific actions, in order of SHAP contribution
    seen = {title}
    for d in top_drivers:
        if d in DRIVER_ACTIONS and DRIVER_ACTIONS[d][0] not in seen:
            t, txt = DRIVER_ACTIONS[d]
            actions.append({"action": t, "detail": txt, "priority": "Medium"})
            seen.add(t)
        if len(actions) >= max_actions:
            break

    # 3. Spend the retention budget where it pays: flag high-value + high-risk
    if band == "High" and row.get("monetary", 0) >= 500 and len(actions) < max_actions + 1:
        actions.insert(0, {"action": "Priority: high value at risk",
                           "detail": f"Lifetime spend {row['monetary']:.0f} and churn risk {p:.0%}: assign to the retention team first.",
                           "priority": "Critical"})
    return actions
