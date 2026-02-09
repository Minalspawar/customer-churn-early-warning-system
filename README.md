# CX Retention & Churn AI Agent (Prototype)

A Streamlit prototype that combines SaaS product signals (usage, subscriptions, churn events, support tickets) with optional LLM-based ticket understanding to identify **at-risk customers**, explain **why** they are at risk, and recommend **actions**.

This project is designed for demos and early experimentation: it produces an executive view, churn drivers, an at-risk action list, and a one-ticket “analyze” tool using a **local Ollama LLM**.

---

## What this app does (in simple terms)

A SaaS business has many signals that hint whether a customer might churn:

- usage is dropping  
- support volume is rising  
- errors are increasing  
- churn events exist for an account  
- ticket text contains frustration or cancellation language (optional, via LLM)

This app loads SaaS CSV tables, calculates those signals, scores each subscription, and shows results in a Streamlit dashboard. Optionally, it uses a local LLM (Ollama) to read ticket text and extract structured insights.

---

## Key features

- **Executive tab**: high-level metrics (churn label rate, high-risk %, ARR stats)
- **Drivers tab**: risk score by plan tier, top churn drivers chart, top-30 risk table, churn reasons from churn_events
- **At-Risk List tab**: filters + customer brief + action list + CSV export
- **Ticket Analyzer tab**: paste any ticket text → LLM returns JSON insights

---

## Datasets used

### SaaS dataset (Ravenstack)
Loaded from `data/raw/saas/` (expected filenames below):

- `ravenstack_accounts.csv`
- `ravenstack_subscriptions.csv`
- `ravenstack_feature_usage.csv`
- `ravenstack_churn_events.csv`
- `ravenstack_support_tickets.csv`

These tables provide:
- account metadata (industry, country, signup date, etc.)
- subscription details (plan tier, MRR/ARR, trial flags, etc.)
- usage events by subscription over time
- churn events (with churn reason codes)
- support tickets (counts + SLA-like fields)

### Ticket dataset (LLM demo)
Loaded from `data/raw/tickets/` as one or more CSV files.

This dataset is used for **LLM text analysis only** (summaries, categories, sentiment, churn intent). If the ticket dataset does not contain `customer_id`, a **demo-only random mapping** to customers is applied.

---

## Project structure

Typical layout:

```
cx-retention-agent/
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
      tickets/
        <one or more ticket CSV files>
  requirements.txt
  README.md
```

### Important files
- **app.py**: Streamlit UI + data loading + feature engineering + scoring + visualizations
- **src/data_loader.py**: helper utilities to list/load CSVs and pick files from folders
- **src/llm_tickets.py**: calls local Ollama and converts ticket text into structured JSON fields
- **src/agent_logic.py**: churn risk scoring logic (`score_customer`)

---

## Setup (Windows + PowerShell)

### 1) Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install Python dependencies
```powershell
pip install -r requirements.txt
```

### 3) Install and run Ollama (local LLM)
1. Install Ollama from the official site: https://ollama.com/
2. Pull a model (example):
```powershell
ollama pull llama3.1:8b
```

3. Quick test:
```powershell
ollama run llama3.1:8b "Return ONLY JSON: {\"ok\": true}"
```

### 4) Run Streamlit
```powershell
streamlit run app.py
```

---

## How to use the app

### Dataset mode dropdown
Located in the sidebar:

- **Ravenstack SaaS (signals + churn label)**  
  Runs the app purely on SaaS tables (usage/support/churn signals). No external ticket LLM dataset required.

- **Tickets only (LLM demo)**  
  Enables ticket dataset selection and LLM processing features (demo). SaaS tables are still loaded because the app’s customer table is built from subscriptions.

- **Both (SaaS + Ticket LLM demo)**  
  Uses SaaS tables + ticket dataset LLM insights together.

### Ticket dataset input (when in Tickets-only or Both modes)
- Choose a ticket CSV from `data/raw/tickets/` (dropdown)
- The selected file is shown in the sidebar as “Tickets main: …”

### LLM processing (button-based)
Controls batch processing of tickets using the local model:

- **Enable LLM ticket extraction**: turns LLM features on/off
- **Tickets to process with LLM**: how many rows to analyze from the ticket dataset
- **Timeout per ticket (seconds)**: safety limit so the app won’t hang on one ticket
- **Run LLM**: starts batch analysis (progress shown)
- **Clear**: clears cached LLM results

LLM results show up under **Drivers** tab (LLM insights section), and can optionally contribute to churn-intent signals.

### Ticket Analyzer tab
Paste any single ticket text and click **Analyze with LLM** to see structured output JSON.

---

## Scoring logic (high-level)

Each subscription becomes a “customer row” with engineered features such as:
- usage_last_30d vs usage_previous_30d → usage_drop_pct
- errors_last_30d
- tickets_last_30d (support load)
- tenure_days
- optional churn_intent from LLM outputs

`score_customer()` combines these signals into:
- **risk_score** (0–100)
- **risk_level** (Low / Medium / High)
- **top_reasons** (explainability)
- **recommended_actions** (playbook actions)

---

## Notes / limitations

- Ticket dataset mapping to customers is **demo-only** unless the ticket file includes `customer_id` that matches the SaaS subscriptions.
- LLM performance depends on hardware; large models may be slow on CPU.
- This prototype is intended for learning, prototyping, and demo usage rather than production.

---

## Troubleshooting

### LLM seems “stuck” on a ticket
- Reduce “Tickets to process with LLM” (e.g., 1–3)
- Reduce model size (try a smaller Ollama model)
- Increase “Timeout per ticket (seconds)”
- Confirm Ollama is running:
```powershell
ollama ps
```

### Tickets dropdown is empty
- Ensure there is at least one CSV inside `data/raw/tickets/`

### Missing SaaS files
- Ensure the 5 Ravenstack CSV files exist in `data/raw/saas/` with exact names expected by `app.py`

---

## License
Add a license of choice (MIT recommended) or remove this section.

---

## Contact / Maintainer
Add team/contact details here.
