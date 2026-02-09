def score_customer(row):
    score = 0
    reasons = []

    usage_drop = float(row.get("usage_drop_pct", 0) or 0)
    tickets = float(row.get("tickets_30d", 0) or 0)
    churn_intent = float(row.get("churn_intent_30d", 0) or 0)
    errors = float(row.get("errors_30d", 0) or 0)
    sat = row.get("satisfaction_score", None)
    escalation = row.get("escalation_flag", 0)
    downgrade = row.get("downgrade_flag", 0)
    is_trial = row.get("is_trial", False)

    # 1) Usage drop (core churn signal)
    if usage_drop >= 0.60:
        score += 40; reasons.append(f"Usage dropped {int(usage_drop*100)}% (30d vs prior)")
    elif usage_drop >= 0.40:
        score += 28; reasons.append(f"Usage dropped {int(usage_drop*100)}% (30d vs prior)")
    elif usage_drop >= 0.20:
        score += 12; reasons.append(f"Usage dropped {int(usage_drop*100)}% (30d vs prior)")

    # 2) Support load
    if tickets >= 6:
        score += 18; reasons.append(f"High support load ({int(tickets)} tickets in 30d)")
    elif tickets >= 3:
        score += 10; reasons.append(f"Elevated support load ({int(tickets)} tickets in 30d)")

    # 3) Product quality / friction (errors)
    if errors >= 10:
        score += 15; reasons.append(f"High error volume ({int(errors)} errors in 30d)")
    elif errors >= 3:
        score += 8; reasons.append(f"Some errors reported ({int(errors)} errors in 30d)")

    # 4) Explicit churn intent from LLM
    if churn_intent >= 1:
        score += 22; reasons.append("Customer language suggests cancellation risk")

    # 5) Support experience quality (if present)
    # satisfaction_score can be missing; handle safely
    try:
        if sat is not None and float(sat) > 0:
            sat = float(sat)
            if sat <= 2:
                score += 12; reasons.append("Low satisfaction score (1–2/5)")
            elif sat == 3:
                score += 6; reasons.append("Medium satisfaction score (3/5)")
    except:
        pass

    # 6) Escalation / downgrade / trial risk
    try:
        if int(escalation) == 1:
            score += 10; reasons.append("Ticket escalation (higher risk)")
    except:
        pass

    try:
        if int(downgrade) == 1:
            score += 12; reasons.append("Recent downgrade (value risk)")
    except:
        pass

    # Trials are naturally higher risk unless activated
    if bool(is_trial):
        score += 8; reasons.append("Trial customer (activation risk)")

    score = int(min(score, 100))

    # IMPORTANT: more realistic thresholds so you actually get High-risk customers
    level = "High" if score >= 60 else ("Medium" if score >= 30 else "Low")

    actions = []
    if any("Usage dropped" in r for r in reasons):
        actions += [
            "Schedule a check-in to reset goals",
            "Recommend 1–2 key features to activate this week",
        ]
    if any("support load" in r.lower() for r in reasons) or any("Ticket escalation" in r for r in reasons):
        actions += [
            "Confirm resolution plan and close the loop",
            "Escalate repeat issues with a clear timeline",
        ]
    if any("cancellation risk" in r.lower() for r in reasons):
        actions += [
            "Offer a value review (ROI + quick wins)",
            "Create a 2-week success plan with milestones",
        ]
    if any("Trial customer" in r for r in reasons):
        actions += [
            "Offer guided onboarding session",
            "Share onboarding checklist + milestones",
        ]
    if any("Low satisfaction" in r for r in reasons):
        actions += [
            "Follow up on dissatisfaction with a recovery plan",
            "Offer proactive support for the next 7 days",
        ]
    if any("downgrade" in r.lower() for r in reasons):
        actions += [
            "Run a value-gap call to understand missing needs",
            "Offer training or enablement for key workflows",
        ]

    if not actions:
        actions = ["Maintain regular touchpoints and monitor health signals"]

    return score, level, "; ".join(reasons[:3]), "; ".join(actions[:3])
