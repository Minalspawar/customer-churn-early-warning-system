import json
import ollama

#MODEL_NAME = "llama3.1:8b"
MODEL_NAME = "llama3.2:3b"


def ticket_to_structured(ticket_text: str) -> dict:
    # keep it small so it doesn't hang
    ticket_text = (ticket_text or "").strip().replace("\x00", "")
    ticket_text = ticket_text[:800]  # smaller = faster

    prompt = (
        "Return ONLY valid JSON. No extra words.\n"
        "Schema:\n"
        '{'
        '"summary": "string (max 20 words)", '
        '"category": "billing|onboarding|bug|feature_request|integration|other", '
        '"sentiment": "positive|neutral|negative", '
        '"churn_intent": true|false, '
        '"recommended_action": "string (max 15 words)"'
        '}\n\n'
        f"Ticket:\n{ticket_text}"
    )

    try:
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_ctx": 1024,      # IMPORTANT: smaller context
                "num_predict": 200,   # IMPORTANT: cap output tokens
                "top_p": 0.9,
            },
        )

        raw = resp["message"]["content"].strip()

        # If model accidentally adds text, try to extract JSON block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]

        return json.loads(raw)

    except Exception as e:
        # Never crash Streamlit
        return {
            "error": str(e),
            "summary": "",
            "category": "other",
            "sentiment": "neutral",
            "churn_intent": False,
            "recommended_action": "",
        }
