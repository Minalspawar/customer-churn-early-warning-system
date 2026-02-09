
# Customer Churn Early-Warning System (CX Retention & Churn AI Agent)

A Streamlit dashboard that helps a Customer Success (CS) team identify **which customers are likely to churn**, **why they look risky**, and **what to do next**. It combines structured SaaS signals (usage, subscriptions, churn events, support tickets) with an **optional local LLM** (Ollama) that can read support-ticket text and extract additional signals (summary, category, sentiment, churn intent).

This project is designed for demos and early experimentation: it produces an executive view, churn drivers, a filterable at-risk action list, and a one-ticket “analyze” tool using a **local Ollama LLM**.

## Project overview

Customers usually show warning signs before they cancel: they use the product less, report more issues, or ask about pricing and renewal. This app pulls those signs together into one dashboard and turns them into an **early-warning action list** so CS teams can reach out proactively.

The “AI agent” part comes from automated decision support:
- it **detects churn risk** from multiple signals,
- **explains the reasons** behind each risk score,
- **recommends next actions** (playbook steps),
- and can use an LLM to **understand ticket language** (e.g., cancellation intent).


## Key features

### Executive tab
High-level health snapshot for leadership:
- churn label rate
- high-risk percentage
- ARR context (average and total)

### Drivers tab
Explains what is driving risk across the customer base:
- **Risk score by plan tier**
- **Top churn drivers** chart (based on triggered “Top Reasons”)
- **Usage + Support signals** table (top 30 customers by risk)
- **Churn reasons** table (from churn events)
- Optional **LLM insights** (category/sentiment/churn intent)

### At-Risk List tab (Action List + Agent Actions)
Operational workspace for CS execution:
- filters (plan tier, risk level, minimum ARR, churn-only)
- customer brief (why at risk + recommended actions)
- action list table + CSV download
- **Agent Actions** (after selecting a customer):
  - **Next Best Action workflow** (step-by-step plan)
  - **Outreach drafts** (email + Slack templates, no LLM needed)
  - **Agent Memory** (session outcome log)

### Ticket Analyzer tab
Paste a single ticket and get structured JSON from the local model:
- summary
- category
- sentiment
- churn intent
- recommended action


## Datasets used

This prototype uses **two categories** of data.

### 1) Structured SaaS dataset (Ravenstack format)
Loaded from `data/raw/saas/` with expected filenames:

- `ravenstack_accounts.csv`
- `ravenstack_subscriptions.csv`
- `ravenstack_feature_usage.csv`
- `ravenstack_churn_events.csv`
- `ravenstack_support_tickets.csv`

These tables represent different business domains (accounts, billing/subscriptions, usage analytics, churn tracking, and support operations). Keeping them separate mirrors real-world systems and makes it easier to update one domain without changing everything else.

### 2) Ticket text dataset (LLM demo, optional)
Loaded from `data/raw/tickets/` using **Folder mode** (dropdown) or **Upload mode** (upload a CSV from the UI).

This dataset is used only for LLM enrichment (turning unstructured text into structured signals).
If the ticket file does not contain a real `customer_id`, the app applies a **demo-only random mapping** to customers so the text-to-signal workflow can still be demonstrated.



## Project structure

Typical layout:


customer-churn-early-warning-system/
app.py
src/
data_loader.py
llm_tickets.py
agent_logic.py
data/
raw/
saas/
ravenstack_accounts.csv
ravenstack_subscriptions.csv
ravenstack_feature_usage.csv
ravenstack_churn_events.csv
ravenstack_support_tickets.csv
tickets/ <one or more ticket CSV files>
.gitkeep
outputs/ <demo exports and screenshots>
docs/
<report/handbook>
requirements.txt
README.md



### Important files
- **app.py**  
  Streamlit UI + data loading + feature engineering + scoring + tabs + agent actions + exports.
- **src/data_loader.py**  
  CSV helpers (listing/loading files).
- **src/llm_tickets.py**  
  Sends ticket text to local Ollama and converts output into structured JSON fields.
- **src/agent_logic.py**  
  Rule-based churn risk scoring (`score_customer`).



## Setup (Windows + PowerShell)

### 1) Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Install and run Ollama (local LLM)

1. Install Ollama: [https://ollama.com/](https://ollama.com/)
2. Pull a model (example):

```powershell
ollama pull llama3.1:3b
```

Optional quality model:

```powershell
ollama pull llama3.1:8b
```

Quick test:

```powershell
ollama run llama3.1:3b "Return ONLY JSON: {\"ok\": true}"
```

### 4) Run Streamlit

```powershell
streamlit run app.py
```

---

## How to use the app

### Dataset mode dropdown

Located in the left sidebar:

* **Ravenstack SaaS (signals + churn label)**
  Runs the dashboard using only the SaaS tables (usage + support + subscriptions + churn events).

* **Tickets only (LLM demo)**
  Enables ticket dataset loading and LLM processing features. SaaS tables are still loaded because customers are derived from subscriptions.

* **Both (SaaS + Ticket LLM demo)**
  Uses SaaS signals plus optional LLM-derived churn intent signals from ticket text.

### “Files detected” section

Shows which datasets are currently loaded:

* **SaaS tables: accounts/subs/usage/churn/support**
* **Tickets main: <selected file name>** (only appears when a ticket file is loaded)

### Ticket dataset input (Folder or Upload)

Shown when “Tickets only” or “Both” is selected.

* **Folder (data/raw/tickets)**: select a CSV from the local folder using a dropdown.
* **Upload CSV**: upload a ticket CSV directly in the UI; it is saved locally and becomes available for processing.

### LLM processing (button-based)

Batch-run ticket extraction using the local model.

* **Enable LLM ticket extraction**: turn ticket enrichment on/off
* **Tickets to process with LLM**: how many ticket rows to analyze (starts from the top of the CSV)
* **Timeout per ticket (seconds)**: safety limit to avoid the app freezing on a slow ticket
* **Run LLM**: begins processing and shows progress
* **Clear**: clears cached LLM results from the session

LLM results are visible in:

* **Drivers tab** → LLM insights section (category/sentiment + sample structured outputs)

### Ticket Analyzer tab

Paste any single ticket and click **Analyze with LLM** to view JSON output.

If output shows:

```json
{"error":"timeout_after_11s", ...}
```

it means the LLM did not return within the selected timeout (not a crash). Increase timeout, reduce ticket size, or use a smaller model.

---

## Scoring logic (high-level)

Each subscription is treated as a “customer row” and receives engineered features such as:

* **usage_30d**: total usage_count in the last 30 days
* **usage_prev30d**: total usage_count in the previous 30 days
* **usage_drop_pct**: (usage_prev30d − usage_30d) / usage_prev30d (when usage_prev30d > 0)
* **errors_30d**: total error_count in last 30 days
* **tickets_30d**: number of support tickets in last 30 days
* **tenure_days**: days since subscription start date
* optional **churn_intent_30d**: derived from LLM ticket enrichment

`score_customer()` combines these signals to produce:

* **risk_score** (0–100)
* **risk_level** (Low / Medium / High)
* **top_reasons** (explainability)
* **recommended_actions** (retention playbook)

---

## “Agent Actions” (At-Risk List tab)

After selecting a customer, the dashboard provides agent-like execution support:

### Next Best Action workflow

A rule-based checklist created from the customer’s risk reasons (usage drop, support issues, onboarding risk, pricing/cancellation language).

### Outreach drafts

Template-based email + Slack drafts created from top reasons and recommended actions (does not require the LLM).

### Agent memory (session)

Stores outreach outcomes and notes in `st.session_state` and displays a small history table during the current session.

---

## Troubleshooting

### LLM is slow / keeps timing out

* Reduce **Tickets to process with LLM** (try 1–3)
* Increase **Timeout per ticket**
* Use smaller model (e.g., `llama3.1:3b`)
* Confirm Ollama is running:

```powershell
ollama ps
```

### Ticket dropdown is empty (Folder mode)

* Ensure at least one `.csv` exists inside `data/raw/tickets/`

### Missing SaaS files

* Ensure the 5 Ravenstack CSV files exist in `data/raw/saas/` with exact names expected

---

## Notes / limitations

* Ticket dataset → customer mapping is **demo-only** unless the ticket file includes a true `customer_id`.
* Rule-based scoring is explainable but not a trained prediction model.
* Local LLM inference depends on hardware; CPU-only machines may require smaller models and higher timeouts.

---

## License

Choose a license (MIT recommended) or remove this section.

---

## Maintainer

Add contact or team details here.

```
::contentReference[oaicite:0]{index=0}
```


