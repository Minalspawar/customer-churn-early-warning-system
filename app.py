import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import threading

from src.data_loader import load_csv
from src.llm_tickets import ticket_to_structured
from src.agent_logic import score_customer

st.set_page_config(page_title="CX Retention & Churn Agent", layout="wide")
st.title("CX Retention & Churn AI Agent (Prototype)")

SAAS_FOLDER = "data/raw/saas"
TICKETS_FOLDER = "data/raw/tickets"
SAAS_UPLOADS_DIR = os.path.join(SAAS_FOLDER, "_uploads")
TICKET_UPLOADS_DIR = os.path.join(TICKETS_FOLDER, "_uploads")

# ----------------------------
# Helpers
# ----------------------------
def require_folder(path: str, name: str):
    if not os.path.exists(path):
        st.error(f"Missing {name} folder: {path}")
        st.stop()

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce") if series is not None else pd.to_datetime([], errors="coerce")

def llm_call_with_timeout(text: str, timeout_s: int = 60) -> dict:
    """
    Hard timeout wrapper so Streamlit never gets stuck on a single ticket.
    """
    result = {
        "error": f"timeout_after_{timeout_s}s",
        "summary": "",
        "category": "other",
        "sentiment": "neutral",
        "churn_intent": False,
        "recommended_action": "",
    }

    def _run():
        nonlocal result
        try:
            out = ticket_to_structured(text)
            if isinstance(out, dict):
                result = out
            else:
                result = {
                    "error": "llm_return_not_dict",
                    "summary": "",
                    "category": "other",
                    "sentiment": "neutral",
                    "churn_intent": False,
                    "recommended_action": "",
                }
        except Exception as e:
            result = {
                "error": str(e),
                "summary": "",
                "category": "other",
                "sentiment": "neutral",
                "churn_intent": False,
                "recommended_action": "",
            }

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result

def build_next_best_action(row: pd.Series) -> dict:
    """
    Rule-based 'agent' workflow generator.
    Returns a structured plan: goal, steps, and follow-up checks.
    """
    reasons = str(row.get("top_reasons", "")).lower()
    plan = {
        "goal": "Reduce churn risk and restore product value",
        "steps": [],
        "follow_up": [],
    }

    # Usage drop
    if "usage dropped" in reasons:
        plan["steps"] += [
            "Schedule a 15–30 minute check-in with key contact",
            "Confirm primary goal + success criteria for next 2 weeks",
            "Recommend 1–2 high-impact features to activate this week",
            "Send a short recap email with agreed next steps",
        ]
        plan["follow_up"] += [
            "Check usage trend after 7 days",
            "Confirm feature activation milestones",
        ]

    # Support load / issues
    if "support" in reasons or "ticket" in reasons or "errors" in reasons:
        plan["steps"] += [
            "Confirm root cause and current resolution status",
            "Escalate repeat issues with target timeline",
            "Share workaround or best-practice guidance",
        ]
        plan["follow_up"] += [
            "Re-check ticket volume and SLA metrics in 7 days",
            "Verify customer satisfaction after resolution",
        ]

    # Onboarding / trial
    if "trial" in reasons or "onboarding" in reasons or "new customer" in reasons:
        plan["steps"] += [
            "Offer guided onboarding session",
            "Share onboarding checklist + milestone timeline",
            "Confirm admin setup and user adoption plan",
        ]
        plan["follow_up"] += [
            "Confirm activation milestones within 3–5 business days",
        ]

    # Cancellation intent / pricing signals
    if "cancellation" in reasons or "cancel" in reasons or "budget" in reasons or "pricing" in reasons:
        plan["steps"] += [
            "Run a value review: ROI + quick wins",
            "Discuss plan fit (tier downgrade vs discount vs scope alignment)",
            "Provide a 2-week success plan with measurable outcomes",
        ]
        plan["follow_up"] += [
            "Confirm decision timeline and renewal date",
            "Track churn intent mentions over next 14 days",
        ]

    if not plan["steps"]:
        plan["steps"] = [
            "Maintain regular touchpoints",
            "Monitor usage, tickets, and account changes weekly",
        ]
        plan["follow_up"] = ["Re-score risk weekly"]

    # De-duplicate while preserving order
    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    plan["steps"] = dedupe(plan["steps"])
    plan["follow_up"] = dedupe(plan["follow_up"])
    return plan

def build_outreach_drafts(row: pd.Series) -> dict:
    """
    Template-based outreach drafts (email + slack) using Top Reasons + Recommended Actions.
    No LLM required.
    """
    account_name = row.get("account_name", "") or "Customer"
    plan = row.get("plan_tier", "")
    arr = int(row.get("arr", 0) or 0)
    reasons = row.get("top_reasons", "")
    actions = row.get("recommended_actions", "")

    email_subject = f"Quick check-in: success plan for {account_name}"
    email_body = (
        f"Hello {account_name} team,\n\n"
        f"A quick check-in regarding the account. Recent signals indicate potential risk:\n"
        f"- {reasons}\n\n"
        f"Suggested next steps:\n"
        f"- {actions}\n\n"
        f"Would a 15–30 minute call this week work to confirm goals and align on a short success plan?\n\n"
        f"Plan: {plan} | ARR: ${arr:,}\n"
        f"Thank you,\n"
        f"Customer Success\n"
    )

    slack_msg = (
        f"Hi {account_name} team — quick check-in.\n"
        f"Recent signals: {reasons}\n"
        f"Proposed next steps: {actions}\n"
        f"Open to a 15–30 min sync this week to align on a short success plan?"
    )

    return {
        "email_subject": email_subject,
        "email_body": email_body,
        "slack_message": slack_msg,
    }

def list_csvs_local(folder: str):
    if not os.path.exists(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(folder, f))
    ]

def save_uploaded_file(uploaded_file, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, uploaded_file.name)
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ----------------------------
# Validate base folders
# ----------------------------
require_folder("data/raw", "data/raw")
require_folder(SAAS_FOLDER, "SaaS (data/raw/saas)")
if not os.path.exists(TICKETS_FOLDER):
    st.warning("Tickets folder missing: data/raw/tickets (Tickets-only mode and LLM demo will be limited)")

# ----------------------------
# Dataset mode
# ----------------------------
dataset_mode = st.sidebar.selectbox(
    "Dataset mode",
    [
        "Ravenstack SaaS (signals + churn label)",
        "Tickets only (LLM demo)",
        "Both (SaaS + Ticket LLM demo)"
    ]
)

st.sidebar.divider()

# ============================================================================
# SaaS dataset input (Folder OR Upload + Mapping)  ✅ NEW
# ============================================================================
st.sidebar.markdown("### SaaS dataset input")

saas_input_mode = st.sidebar.radio(
    "Load SaaS data from",
    ["Folder (data/raw/saas)", "Upload SaaS CSVs"],
    horizontal=False
)

# Default Ravenstack expected filenames (folder mode convenience)
DEFAULT_FILES = {
    "accounts": os.path.join(SAAS_FOLDER, "ravenstack_accounts.csv"),
    "subscriptions": os.path.join(SAAS_FOLDER, "ravenstack_subscriptions.csv"),
    "usage": os.path.join(SAAS_FOLDER, "ravenstack_feature_usage.csv"),
    "churn": os.path.join(SAAS_FOLDER, "ravenstack_churn_events.csv"),
    "support": os.path.join(SAAS_FOLDER, "ravenstack_support_tickets.csv"),
}

saas_file_map = {"accounts": None, "subscriptions": None, "usage": None, "churn": None, "support": None}
saas_selected_names = {}

if saas_input_mode == "Folder (data/raw/saas)":
    # Use the Ravenstack default names if present; allow missing optional files.
    for k, p in DEFAULT_FILES.items():
        if os.path.exists(p):
            saas_file_map[k] = p

    # Hard requirement: subscriptions + usage must exist to build risk signals
    if not saas_file_map["subscriptions"] or not saas_file_map["usage"]:
        st.error(
            "Missing required SaaS files in folder. At minimum these must exist:\n"
            "- ravenstack_subscriptions.csv\n"
            "- ravenstack_feature_usage.csv\n\n"
            "Switch to 'Upload SaaS CSVs' if filenames are different."
        )
        st.stop()

    for k, p in saas_file_map.items():
        if p:
            saas_selected_names[k] = os.path.basename(p)

else:
    st.sidebar.caption("Upload 2+ CSVs. Minimum required: Subscriptions + Usage.")
    uploaded_saas = st.sidebar.file_uploader(
        "Upload SaaS CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    uploaded_paths = []
    if uploaded_saas:
        for f in uploaded_saas:
            saved = save_uploaded_file(f, SAAS_UPLOADS_DIR)
            uploaded_paths.append(saved)

    if not uploaded_paths:
        st.sidebar.info("Upload SaaS CSVs to continue.")
        st.stop()

    # Mapping UI: pick which file is which table
    uploaded_names = [os.path.basename(p) for p in uploaded_paths]
    name_to_path = {os.path.basename(p): p for p in uploaded_paths}

    st.sidebar.markdown("#### Map uploaded files to tables")
    st.sidebar.caption("Subscriptions + Usage are required. Others are optional.")

    def pick_file(label: str, required: bool):
        options = ["(Not provided)"] + uploaded_names
        idx = 0
        val = st.sidebar.selectbox(
            label,
            options,
            index=idx
        )
        if val == "(Not provided)":
            if required:
                st.sidebar.warning(f"{label} is required.")
            return None
        return name_to_path[val]

    saas_file_map["subscriptions"] = pick_file("Subscriptions CSV (required)", required=True)
    saas_file_map["usage"] = pick_file("Usage CSV (required)", required=True)
    saas_file_map["accounts"] = pick_file("Accounts CSV (optional)", required=False)
    saas_file_map["churn"] = pick_file("Churn events CSV (optional)", required=False)
    saas_file_map["support"] = pick_file("Support tickets CSV (optional)", required=False)

    if not saas_file_map["subscriptions"] or not saas_file_map["usage"]:
        st.error("Please map BOTH Subscriptions and Usage CSVs to continue.")
        st.stop()

    for k, p in saas_file_map.items():
        if p:
            saas_selected_names[k] = os.path.basename(p)

st.sidebar.divider()

# ----------------------------
# Load SaaS data (flexible)
# ----------------------------
subs_df = load_csv(saas_file_map["subscriptions"])
usage_df = load_csv(saas_file_map["usage"])

accounts_df = load_csv(saas_file_map["accounts"]) if saas_file_map["accounts"] else pd.DataFrame()
churn_df = load_csv(saas_file_map["churn"]) if saas_file_map["churn"] else pd.DataFrame()
saas_tickets_df = load_csv(saas_file_map["support"]) if saas_file_map["support"] else pd.DataFrame()

# ----------------------------
# Normalize & ensure minimum columns
# ----------------------------
# Required columns in subs + usage
subs_df = ensure_cols(subs_df, ["subscription_id", "account_id", "plan_tier", "mrr_amount", "arr_amount", "start_date", "end_date"])
usage_df = ensure_cols(usage_df, ["subscription_id", "usage_date", "usage_count", "error_count"])

# Optional tables: create empty with expected columns so downstream works
if accounts_df.empty:
    accounts_df = pd.DataFrame(columns=["account_id", "account_name", "industry", "country", "signup_date", "referral_source"])
else:
    accounts_df = ensure_cols(accounts_df, ["account_id", "account_name", "industry", "country", "signup_date", "referral_source"])

if churn_df.empty:
    churn_df = pd.DataFrame(columns=["account_id", "churn_date", "reason_code"])
else:
    churn_df = ensure_cols(churn_df, ["account_id", "churn_date", "reason_code"])

if saas_tickets_df.empty:
    saas_tickets_df = pd.DataFrame(columns=["account_id", "submitted_at", "first_response_time_minutes", "resolution_time_hours", "escalation_flag", "satisfaction_score", "closed_at"])
else:
    saas_tickets_df = ensure_cols(saas_tickets_df, ["account_id", "submitted_at", "first_response_time_minutes", "resolution_time_hours", "escalation_flag", "satisfaction_score", "closed_at"])

# Type conversions
subs_df = subs_df.copy()
usage_df = usage_df.copy()
accounts_df = accounts_df.copy()
churn_df = churn_df.copy()
saas_tickets_df = saas_tickets_df.copy()

subs_df["subscription_id"] = subs_df["subscription_id"].astype(str)
subs_df["account_id"] = subs_df["account_id"].astype(str)
subs_df["start_date"] = safe_to_datetime(subs_df.get("start_date"))
subs_df["end_date"] = safe_to_datetime(subs_df.get("end_date"))

usage_df["subscription_id"] = usage_df["subscription_id"].astype(str)
usage_df["usage_date"] = safe_to_datetime(usage_df.get("usage_date"))
usage_df["usage_count"] = pd.to_numeric(usage_df.get("usage_count"), errors="coerce").fillna(0)
usage_df["error_count"] = pd.to_numeric(usage_df.get("error_count"), errors="coerce").fillna(0)

accounts_df["account_id"] = accounts_df["account_id"].astype(str)
accounts_df["signup_date"] = safe_to_datetime(accounts_df.get("signup_date"))

churn_df["account_id"] = churn_df["account_id"].astype(str)
churn_df["churn_date"] = safe_to_datetime(churn_df.get("churn_date"))

saas_tickets_df["account_id"] = saas_tickets_df["account_id"].astype(str)
saas_tickets_df["submitted_at"] = safe_to_datetime(saas_tickets_df.get("submitted_at"))
if "closed_at" in saas_tickets_df.columns:
    saas_tickets_df["closed_at"] = safe_to_datetime(saas_tickets_df.get("closed_at"))

# ----------------------------
# Debug panes (SaaS)
# ----------------------------
with st.expander("Debug: SaaS table columns"):
    st.write("accounts:", list(accounts_df.columns))
    st.write("subscriptions:", list(subs_df.columns))
    st.write("usage:", list(usage_df.columns))
    st.write("churn:", list(churn_df.columns))
    st.write("support_tickets:", list(saas_tickets_df.columns))

# ----------------------------
# Build customer table (subscription-level) + account-level context
# ----------------------------
customers = subs_df.copy().rename(columns={"subscription_id": "customer_id"})
customers["customer_id"] = customers["customer_id"].astype(str)

keep_acct_cols = ["account_id", "account_name", "industry", "country", "signup_date", "referral_source"]
acct_cols = [c for c in keep_acct_cols if c in accounts_df.columns]
if acct_cols and not accounts_df.empty:
    customers = customers.merge(accounts_df[acct_cols], on="account_id", how="left")

# ARR + plan tier
customers["arr"] = pd.to_numeric(customers["arr_amount"], errors="coerce").fillna(0)
customers["mrr"] = pd.to_numeric(customers["mrr_amount"], errors="coerce").fillna(0)
customers["plan_tier"] = customers["plan_tier"].astype(str).fillna("Unknown")

# Tenure (days since start_date, anchored to latest usage date)
anchor_date = usage_df["usage_date"].max()
if pd.isna(anchor_date):
    anchor_date = pd.Timestamp.today()
customers["tenure_days"] = (anchor_date - customers["start_date"]).dt.days
customers["tenure_days"] = customers["tenure_days"].fillna(0).clip(lower=0)

# ----------------------------
# Usage metrics: last 30d vs previous 30d
# ----------------------------
end_date = usage_df["usage_date"].max()
if pd.isna(end_date):
    end_date = pd.Timestamp.today()

last30_start = end_date - pd.Timedelta(days=30)
prev30_start = end_date - pd.Timedelta(days=60)

u_last30 = (
    usage_df[(usage_df["usage_date"] >= last30_start) & (usage_df["usage_date"] <= end_date)]
    .groupby("subscription_id")["usage_count"].sum()
    .rename("usage_30d")
    .reset_index()
)

u_prev30 = (
    usage_df[(usage_df["usage_date"] >= prev30_start) & (usage_df["usage_date"] < last30_start)]
    .groupby("subscription_id")["usage_count"].sum()
    .rename("usage_prev30d")
    .reset_index()
)

err_last30 = (
    usage_df[(usage_df["usage_date"] >= last30_start) & (usage_df["usage_date"] <= end_date)]
    .groupby("subscription_id")["error_count"].sum()
    .rename("errors_30d")
    .reset_index()
)

customers = customers.merge(u_last30, left_on="customer_id", right_on="subscription_id", how="left").drop(columns=["subscription_id"])
customers = customers.merge(u_prev30, left_on="customer_id", right_on="subscription_id", how="left").drop(columns=["subscription_id"])
customers = customers.merge(err_last30, left_on="customer_id", right_on="subscription_id", how="left").drop(columns=["subscription_id"])

customers["usage_30d"] = customers["usage_30d"].fillna(0)
customers["usage_prev30d"] = customers["usage_prev30d"].fillna(0)
customers["errors_30d"] = customers["errors_30d"].fillna(0)

customers["usage_drop_pct"] = np.where(
    customers["usage_prev30d"] > 0,
    (customers["usage_prev30d"] - customers["usage_30d"]) / customers["usage_prev30d"],
    0.0
)
customers["usage_drop_pct"] = np.clip(customers["usage_drop_pct"], 0, None)

# ----------------------------
# Churn label (optional account-level churn events)
# ----------------------------
if not churn_df.empty and "account_id" in churn_df.columns:
    customers["churned"] = customers["account_id"].isin(set(churn_df["account_id"].dropna().astype(str)))
else:
    customers["churned"] = False

# ----------------------------
# Support metrics (optional, last 30 days) by account_id
# ----------------------------
if not saas_tickets_df.empty and "submitted_at" in saas_tickets_df.columns:
    ticket_end = saas_tickets_df["submitted_at"].max()
    if pd.isna(ticket_end):
        ticket_end = pd.Timestamp.today()
    ticket_last30 = ticket_end - pd.Timedelta(days=30)

    t30 = saas_tickets_df[(saas_tickets_df["submitted_at"] >= ticket_last30) & (saas_tickets_df["submitted_at"] <= ticket_end)].copy()
    tickets_30d = t30.groupby("account_id").size().rename("tickets_30d").reset_index()

    customers = customers.merge(tickets_30d, on="account_id", how="left")
    customers["tickets_30d"] = customers["tickets_30d"].fillna(0)

    # Optional SLA metrics
    agg_map = {}
    if "first_response_time_minutes" in t30.columns:
        agg_map["avg_first_response_min"] = ("first_response_time_minutes", "mean")
    if "resolution_time_hours" in t30.columns:
        agg_map["avg_resolution_hrs"] = ("resolution_time_hours", "mean")
    if "escalation_flag" in t30.columns:
        agg_map["escalations_30d"] = ("escalation_flag", "sum")
    if "satisfaction_score" in t30.columns:
        agg_map["avg_csat"] = ("satisfaction_score", "mean")

    if agg_map:
        sla_df = t30.groupby("account_id").agg(**agg_map).reset_index()
        customers = customers.merge(sla_df, on="account_id", how="left")
        for c in ["avg_first_response_min", "avg_resolution_hrs", "escalations_30d", "avg_csat"]:
            if c in customers.columns:
                customers[c] = pd.to_numeric(customers[c], errors="coerce").fillna(0)
else:
    ticket_end = pd.Timestamp.today()
    customers["tickets_30d"] = 0

# ============================================================================
# Ticket dataset input (Folder or Upload) - (unchanged from your version)
# ============================================================================
tickets_df = None
text_col = None
ticket_file_selected = None

if dataset_mode in ["Tickets only (LLM demo)", "Both (SaaS + Ticket LLM demo)"]:

    st.sidebar.markdown("### Ticket dataset input")
    ticket_input_mode = st.sidebar.radio(
        "Load ticket data from",
        ["Folder (data/raw/tickets)", "Upload CSV"],
        horizontal=False
    )

    tickets_main = None

    if ticket_input_mode == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload ticket CSV", type=["csv"])
        if uploaded is not None:
            tickets_main = save_uploaded_file(uploaded, TICKET_UPLOADS_DIR)
            ticket_file_selected = os.path.basename(tickets_main)
        else:
            st.sidebar.info("Upload a CSV to enable ticket processing.")
    else:
        if not os.path.exists(TICKETS_FOLDER):
            st.error("Tickets folder not found. Create data/raw/tickets and unzip a ticket dataset there.")
            st.stop()

        ticket_csvs = list_csvs_local(TICKETS_FOLDER)
        if not ticket_csvs:
            st.error("No Ticket CSV files found in data/raw/tickets.")
            st.stop()

        ticket_files = sorted([os.path.basename(p) for p in ticket_csvs])
        ticket_file_selected = st.sidebar.selectbox("Ticket dataset file", ticket_files)
        tickets_main = os.path.join(TICKETS_FOLDER, ticket_file_selected)

    if tickets_main:
        tickets_df = load_csv(tickets_main)

        with st.expander("Debug: Ticket columns"):
            st.write(list(tickets_df.columns))
            st.dataframe(tickets_df.head(5), use_container_width=True)

        tickets_df = tickets_df.copy()

        if ("subject" in tickets_df.columns) and ("body" in tickets_df.columns):
            # SAFE: keep input smaller (helps speed)
            tickets_df["ticket_text_for_llm"] = (
                tickets_df["subject"].astype(str).str.slice(0, 300)
                + "\n\n"
                + tickets_df["body"].astype(str).str.replace("\x00", "", regex=False).str.slice(0, 800)
            )
            text_col = "ticket_text_for_llm"
        else:
            candidate_cols = [c for c in tickets_df.columns if str(c).lower() in ["ticket_text", "text", "body", "message", "content", "description"]]
            if candidate_cols:
                text_col = candidate_cols[0]
            else:
                obj_cols = tickets_df.select_dtypes(include="object").columns.tolist()
                text_col = obj_cols[0] if obj_cols else tickets_df.columns[0]

        # Demo mapping: external ticket dataset has no real customer_id, so we random-map.
        if "customer_id" not in tickets_df.columns:
            tickets_df["customer_id"] = np.random.choice(customers["customer_id"].astype(str).tolist(), size=len(tickets_df), replace=True)
        tickets_df["customer_id"] = tickets_df["customer_id"].astype(str)

# ----------------------------
# Sidebar header (files detected)
# ----------------------------
st.sidebar.success("Files detected")
st.sidebar.write("SaaS tables loaded:")
st.sidebar.write(f"- Subscriptions: {saas_selected_names.get('subscriptions','(mapped)')}")
st.sidebar.write(f"- Usage: {saas_selected_names.get('usage','(mapped)')}")
st.sidebar.write(f"- Accounts: {saas_selected_names.get('accounts','(not provided)')}")
st.sidebar.write(f"- Churn: {saas_selected_names.get('churn','(not provided)')}")
st.sidebar.write(f"- Support: {saas_selected_names.get('support','(not provided)')}")
if ticket_file_selected:
    st.sidebar.write("Tickets main:", ticket_file_selected)

# ----------------------------
# LLM processing (button-based)
# ----------------------------
st.sidebar.markdown("### LLM processing (button-based)")

if "enriched" not in st.session_state:
    st.session_state["enriched"] = pd.DataFrame([])
if "llm_last_run" not in st.session_state:
    st.session_state["llm_last_run"] = None

enable_llm = st.sidebar.checkbox("Enable LLM ticket extraction", value=True)
n_llm = st.sidebar.slider("Tickets to process with LLM", 1, 30, 5)
llm_timeout_s = st.sidebar.slider("Timeout per ticket (seconds)", 10, 120, 60)

def llm_process(df, n, col, timeout_s: int):
    df = df.head(n).copy()
    rows = []

    prog = st.progress(0)
    status = st.empty()
    st.sidebar.caption(f"LLM column: {col}")

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        txt = str(r[col]) if pd.notna(r[col]) else ""
        txt = txt.replace("\x00", "").strip()

        # extra safety: keep prompt size small + fast
        txt_for_llm = txt[:900]
        preview = (txt_for_llm[:120] + "...") if len(txt_for_llm) > 120 else txt_for_llm

        status.info(f"LLM running... {i}/{n} | chars={len(txt_for_llm)} | timeout={timeout_s}s")
        st.sidebar.caption(f"{i}/{n} preview: {preview}")

        out = llm_call_with_timeout(txt_for_llm, timeout_s=timeout_s)
        out["customer_id"] = str(r["customer_id"])
        out["ticket_chars"] = len(txt_for_llm)
        rows.append(out)

        prog.progress(i / n)

    status.success(f"✅ LLM complete: processed {n}/{n} tickets")
    st.toast("LLM finished ✅", icon="✅")
    return pd.DataFrame(rows)

c_run, c_clear = st.sidebar.columns(2)
run_llm = c_run.button("Run LLM")
clear_llm = c_clear.button("Clear")

if clear_llm:
    st.session_state["enriched"] = pd.DataFrame([])
    st.session_state["llm_last_run"] = None
    st.sidebar.info("Cleared LLM results. Click Run LLM again to reprocess.")

if enable_llm and run_llm:
    if tickets_df is None:
        st.sidebar.error("No ticket dataset loaded (select Tickets-only or Both).")
    else:
        st.session_state["enriched"] = llm_process(tickets_df, n_llm, text_col, timeout_s=llm_timeout_s)
        st.session_state["llm_last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")

if st.session_state["llm_last_run"]:
    st.sidebar.caption(f"Last LLM run: {st.session_state['llm_last_run']}")

enriched = st.session_state["enriched"]

# LLM error counter (safe)
if (enriched is not None) and (not enriched.empty) and ("error" in enriched.columns):
    err_n = (enriched["error"].fillna("").astype(str).str.len() > 0).sum()
    if err_n > 0:
        st.sidebar.warning(f"LLM errors: {err_n}")

# Add churn intent signal to customers
if (enriched is not None) and (not enriched.empty) and ("churn_intent" in enriched.columns):
    churn_intent = (
        enriched.groupby("customer_id")["churn_intent"]
        .sum()
        .rename("churn_intent_30d")
        .reset_index()
    )
    customers = customers.merge(churn_intent, on="customer_id", how="left")
    customers["churn_intent_30d"] = customers["churn_intent_30d"].fillna(0)
else:
    customers["churn_intent_30d"] = 0

# ----------------------------
# Score customers (existing logic)
# ----------------------------
scores = customers.apply(lambda r: score_customer(r), axis=1, result_type="expand")
customers["risk_score"] = scores[0]
customers["risk_level"] = scores[1]
customers["top_reasons"] = scores[2]
customers["recommended_actions"] = scores[3]
customers["churn_probability"] = customers["risk_score"] / 100.0

# ----------------------------
# UI Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Executive", "Drivers", "At-Risk List", "Ticket Analyzer"])

with tab1:
    st.subheader("Executive Summary")

    high_risk_pct = (customers["risk_level"].eq("High").mean() * 100).round(1)
    churn_rate = (customers["churned"].mean() * 100).round(1)
    avg_arr = customers["arr"].mean().round(0)
    total_arr = customers["arr"].sum().round(0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn rate (label)", f"{churn_rate}%")
    c2.metric("High-risk customers", f"{high_risk_pct}%")
    c3.metric("Avg ARR", f"${int(avg_arr):,}")
    c4.metric("Total ARR (dataset)", f"${int(total_arr):,}")

    st.markdown("### What this tool does")
    st.write(
        "- Scores each subscription for churn risk using **usage drop**, **errors**, **support load**, and optional **LLM churn intent**.\n"
        "- Produces an **action list** for CSMs/Onboarders: who to contact first and what to do.\n"
        "- Adds **explainability** via Top Reasons + Recommended Actions."
    )

    st.caption(
        f"Usage window end date: {str(pd.to_datetime(end_date).date())} | "
        f"Support window end date: {str(pd.to_datetime(ticket_end).date())}"
    )

    if dataset_mode == "Tickets only (LLM demo)":
        st.warning(
            "Tickets-only mode is a demo: churn label and SaaS signals come from the selected SaaS dataset, "
            "but ticket-to-customer mapping is random (not real)."
        )

with tab2:
    st.subheader("Drivers & Insights")

    st.markdown("### Risk score by plan tier")
    st.dataframe(customers.groupby("plan_tier")["risk_score"].mean().reset_index(), use_container_width=True)

    st.markdown("### Top churn drivers (from Top Reasons)")
    reasons = customers["top_reasons"].fillna("").astype(str).str.split(";")
    reason_counts = reasons.explode().str.strip()
    reason_counts = reason_counts[reason_counts != ""].value_counts().head(10).reset_index()
    reason_counts.columns = ["reason", "count"]
    if len(reason_counts):
        st.bar_chart(reason_counts.set_index("reason"))
    else:
        st.info("No top reasons available yet. (Check score_customer output.)")

    st.markdown("### Usage + Support signals (top 30 by risk)")
    st.dataframe(
        customers[[
            "customer_id", "account_id", "plan_tier", "arr",
            "usage_30d", "usage_prev30d", "usage_drop_pct",
            "errors_30d", "tickets_30d",
            "risk_score", "risk_level"
        ]]
        .sort_values("risk_score", ascending=False)
        .head(30),
        use_container_width=True,
    )

    st.markdown("### Churn reasons (from churn events)")
    if not churn_df.empty and "reason_code" in churn_df.columns:
        churn_reason = churn_df["reason_code"].fillna("unknown").value_counts().head(10).reset_index()
        churn_reason.columns = ["reason_code", "count"]
        st.dataframe(churn_reason, use_container_width=True)
    else:
        st.info("No churn events provided (or no reason_code column).")

    if enriched is not None and not enriched.empty:
        st.markdown("### LLM insights (sample tickets)")
        st.info("Note: These LLM insights are based on the external ticket dataset and mapped randomly to customers (demo signal).")

        cA, cB = st.columns(2)
        with cA:
            if "category" in enriched.columns:
                cat = enriched["category"].fillna("other").value_counts().reset_index()
                cat.columns = ["category", "count"]
                st.dataframe(cat, use_container_width=True)
        with cB:
            if "sentiment" in enriched.columns:
                sent = enriched["sentiment"].fillna("neutral").value_counts().reset_index()
                sent.columns = ["sentiment", "count"]
                st.dataframe(sent, use_container_width=True)

        cols_show = [c for c in ["summary", "category", "sentiment", "churn_intent", "recommended_action", "customer_id", "ticket_chars", "error"] if c in enriched.columns]
        st.markdown("**Example LLM outputs (first 10)**")
        st.dataframe(enriched[cols_show].head(10), use_container_width=True)

with tab3:
    st.subheader("At-Risk Customers (Action List)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tiers = st.multiselect("Plan tier", sorted(customers["plan_tier"].unique()), default=sorted(customers["plan_tier"].unique()))
    with c2:
        risk = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium"])
    with c3:
        min_arr = st.number_input("Min ARR", value=0)
    with c4:
        only_churned = st.checkbox("Only churned (label)", value=False)

    view = customers[
        (customers["plan_tier"].isin(tiers)) &
        (customers["risk_level"].isin(risk)) &
        (customers["arr"] >= min_arr)
    ].copy()

    if only_churned:
        view = view[view["churned"] == True]

    view = view.sort_values(["risk_score", "arr"], ascending=False)

    st.markdown("### Customer Brief (CSM-ready)")
    show_list = view.head(200)
    pick_list = show_list["customer_id"].astype(str).tolist()

    if pick_list:
        selected = st.selectbox("Pick a customer_id", pick_list)
        row = show_list[show_list["customer_id"].astype(str) == str(selected)].iloc[0]

        name = row.get("account_name", "")
        st.write(f"**Customer:** {row['customer_id']}  |  **Account:** {row['account_id']} {(' | ' + str(name)) if name else ''}")
        st.write(f"**Plan:** {row['plan_tier']}  |  **ARR:** ${int(row['arr']):,}  |  **Churn label:** {row['churned']}")
        st.write(f"**Risk:** {row['risk_level']} ({row['risk_score']}/100)")
        st.write(f"**Why at risk:** {row.get('top_reasons','')}")
        st.write(f"**Recommended actions:** {row.get('recommended_actions','')}")

        st.markdown("---")
        st.markdown("## Agent Actions (Next Best Steps + Outreach + Memory)")

        # Upgrade A: Next Best Action workflow
        with st.expander("✅ Next Best Action workflow (agent plan)", expanded=True):
            plan = build_next_best_action(row)
            st.write(f"**Goal:** {plan['goal']}")
            st.write("**Suggested steps:**")
            for idx, step in enumerate(plan["steps"], start=1):
                st.write(f"{idx}. {step}")
            st.write("**Follow-up checks:**")
            for idx, step in enumerate(plan["follow_up"], start=1):
                st.write(f"{idx}. {step}")

        # Upgrade B: Outreach drafts
        with st.expander("✉️ Outreach drafts (email + Slack)", expanded=False):
            drafts = build_outreach_drafts(row)
            st.text_input("Email subject", value=drafts["email_subject"], key=f"email_subj_{row['customer_id']}")
            st.text_area("Email body", value=drafts["email_body"], height=220, key=f"email_body_{row['customer_id']}")
            st.text_area("Slack message", value=drafts["slack_message"], height=120, key=f"slack_{row['customer_id']}")

        # Upgrade C: Agent memory (session)
        with st.expander("🧠 Agent memory (outcome log)", expanded=False):
            if "agent_memory" not in st.session_state:
                st.session_state["agent_memory"] = []

            outcome = st.selectbox(
                "Outcome after outreach",
                ["No action yet", "Contacted", "Meeting scheduled", "Issue resolved", "Renewed", "Downgraded", "Cancelled", "No response"],
                key=f"outcome_{row['customer_id']}",
            )
            note = st.text_area("Notes (optional)", height=100, key=f"note_{row['customer_id']}")

            cA, cB = st.columns(2)
            if cA.button("Save outcome", key=f"save_{row['customer_id']}"):
                st.session_state["agent_memory"].append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_id": str(row["customer_id"]),
                    "account_id": str(row["account_id"]),
                    "account_name": str(row.get("account_name", "")),
                    "risk_level": str(row.get("risk_level", "")),
                    "risk_score": int(row.get("risk_score", 0) or 0),
                    "outcome": outcome,
                    "note": note,
                })
                st.success("Saved to session memory (visible below).")

            if cB.button("Clear memory", key="clear_agent_memory"):
                st.session_state["agent_memory"] = []
                st.info("Session memory cleared.")

            mem_df = pd.DataFrame(st.session_state["agent_memory"])
            if not mem_df.empty:
                st.dataframe(mem_df.tail(20), use_container_width=True)
            else:
                st.caption("No outcomes saved yet.")
    else:
        st.info("No customers match the selected filters.")

    st.markdown("### Action List")
    cols = [
        "customer_id", "account_id",
        "account_name" if "account_name" in view.columns else None,
        "plan_tier", "arr", "churned",
        "risk_level", "risk_score",
        "tickets_30d", "churn_intent_30d",
        "usage_30d", "usage_prev30d", "usage_drop_pct", "errors_30d",
        "avg_first_response_min" if "avg_first_response_min" in view.columns else None,
        "avg_resolution_hrs" if "avg_resolution_hrs" in view.columns else None,
        "escalations_30d" if "escalations_30d" in view.columns else None,
        "avg_csat" if "avg_csat" in view.columns else None,
        "top_reasons", "recommended_actions",
    ]
    cols = [c for c in cols if c is not None and c in view.columns]

    st.dataframe(view[cols].head(50), use_container_width=True)
    st.download_button("Download at-risk list CSV", view[cols].to_csv(index=False), "at_risk_customers.csv", "text/csv")

with tab4:
    st.subheader("Ticket Analyzer (Local LLM)")
    st.write("Paste any ticket text below and click Analyze. This uses the local Ollama model.")
    ticket = st.text_area("Ticket text", height=160)
    if st.button("Analyze with LLM"):
        out = llm_call_with_timeout(ticket[:900], timeout_s=llm_timeout_s)
        st.json(out)
