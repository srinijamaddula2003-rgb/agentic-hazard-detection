from pathlib import Path
import json
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------- paths ----------
GOLD_PATH = Path("data/processed/annotation_sample.csv")  # your labeled file
OUT_PATH = Path("data/processed/openai_predictions.jsonl")

# ---------- setup ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

# Keep taxonomy small + consistent
ALLOWED_TYPES = [
    "ordering_violation",
    "state_violation",
    "exception_handling",
    "context_reset",
    "credential_handling",
    "other",
]

def build_messages(paragraph: str):
    system = (
        "You are a telecom security analyst. "
        "You must output ONLY valid JSON with the required keys and no extra text."
    )

    user = f"""
A hazard indicator is a conditional statement in LTE/3GPP specifications that links an event/message
to a risky or security-relevant operation, such as:
- aborting a procedure,
- discarding a message,
- clearing/resetting context,
- releasing resources,
- invalidating credentials,
- rejecting an operation.

Given the LTE specification paragraph below, extract ONE main hazard indicator if present.

Paragraph:
\"\"\"{paragraph}\"\"\"

Return a SINGLE JSON object with EXACTLY these keys:
- is_hazard: "yes" or "no"
- condition: string ("" if none)
- operation: string ("" if none)
- state: string ("" if unclear)
- hazard_type: one of {ALLOWED_TYPES}

If there is no hazard indicator, return:
{{"is_hazard":"no","condition":"","operation":"","state":"","hazard_type":""}}
"""
    return system, user

def call_openai(paragraph: str) -> dict:
    system, user = build_messages(paragraph)

    # Use a model that supports JSON output well.
    # If your account has GPT-4.1 / GPT-4o / etc, you can change model name.
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # switch to your preferred model if needed
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content.strip()
    try:
        data = json.loads(text)
    except Exception:
        # fallback
        data = {"is_hazard": "no", "condition": "", "operation": "", "state": "", "hazard_type": ""}

    # Ensure keys exist
    for k in ["is_hazard", "condition", "operation", "state", "hazard_type"]:
        data.setdefault(k, "")

    # Normalize
    data["is_hazard"] = str(data["is_hazard"]).strip().lower()
    if data["is_hazard"] not in ["yes", "no"]:
        data["is_hazard"] = "no"

    ht = str(data["hazard_type"]).strip()
    if data["is_hazard"] == "no":
        data["hazard_type"] = ""
        data["condition"] = ""
        data["operation"] = ""
        data["state"] = ""
    else:
        if ht not in ALLOWED_TYPES:
            data["hazard_type"] = "other"

    return data

def main():
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold file not found: {GOLD_PATH}")

    df = pd.read_csv(GOLD_PATH)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pid = int(row["id"]) if "id" in row and not pd.isna(row["id"]) else None
            paragraph = str(row["paragraph"])

            pred = call_openai(paragraph)

            record = {
                "id": pid,
                "paragraph": paragraph,
                "openai_is_hazard": pred["is_hazard"],
                "openai_condition": pred["condition"],
                "openai_operation": pred["operation"],
                "openai_state": pred["state"],
                "openai_hazard_type": pred["hazard_type"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {OUT_PATH}")

if __name__ == "__main__":
    main()