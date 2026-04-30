from pathlib import Path
import json
import requests
import csv
import time

GOLD_PATH = Path("data/processed/annotation_sample.csv")
PRED_PATH = Path("data/processed/ollama_predictions.jsonl")
OUT_PATH = Path("data/processed/hazard_indicators_structured.jsonl")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

# Known protocol states from TS 24.301
EMM_STATES = [
    "EMM-NULL", "EMM-DEREGISTERED", "EMM-REGISTERED-INITIATED",
    "EMM-REGISTERED", "EMM-DEREGISTERED-INITIATED",
    "EMM-TRACKING-AREA-UPDATING-INITIATED", "EMM-SERVICE-REQUEST-INITIATED",
    "EMM-COMMON-PROCEDURE-INITIATED",
]

ESM_STATES = [
    "BEARER CONTEXT INACTIVE", "BEARER CONTEXT ACTIVE",
    "BEARER CONTEXT ACTIVE PENDING", "BEARER CONTEXT INACTIVE PENDING",
    "BEARER CONTEXT MODIFY PENDING",
]

KNOWN_TIMERS = [
    "T3410", "T3412", "T3417", "T3418", "T3421", "T3422",
    "T3430", "T3450", "T3460", "T3470",
]


def build_recovery_prompt(paragraph, condition, operation):
    system = (
        "You are an LTE protocol analyst. "
        "Return ONLY valid JSON. No extra text."
    )
    user = f"""
Extract state and event details from this LTE hazard indicator.

Paragraph:
\"\"\"{paragraph[:500]}\"\"\"

Already known:
- Condition: {condition}
- Operation: {operation}

Return ONLY this JSON:
{{
  "trigger_message": "exact NAS message name, e.g. AUTHENTICATION REJECT",
  "trigger_direction": "network->UE" or "UE->network" or "internal",
  "entity": "UE" or "MME" or "both",
  "pre_state": "state BEFORE the action",
  "post_state": "state AFTER the action",
  "operations": ["list", "each", "action", "separately"],
  "affected_contexts": ["e.g. EMM context", "EPS bearer context"],
  "timer_impacts": ["e.g. stop T3410", "start T3421"]
}}

Use "" for unknown strings, [] for unknown lists.
"""
    return system, user


def call_ollama(system, user):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 300}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=900)
    r.raise_for_status()
    text = r.json()["message"]["content"].strip()

    try:
        return json.loads(text)
    except:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                return {}
        return {}


def validate_states(pre_state, post_state, timer_impacts):
    all_states = EMM_STATES + ESM_STATES
    warnings = []
    if pre_state and pre_state not in all_states:
        warnings.append(f"Unknown pre_state: {pre_state}")
    if post_state and post_state not in all_states:
        warnings.append(f"Unknown post_state: {post_state}")
    for t in timer_impacts:
        if not any(known in t for known in KNOWN_TIMERS):
            if t:
                warnings.append(f"Unknown timer: {t}")
    return warnings


def main():
    # Load gold annotations for the confirmed HIs
    gold_his = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r['is_hazard'].strip().lower() == 'yes':
                gold_his.append(r)

    # Load predictions to get LLM-extracted fields for matched HIs
    preds = {}
    if PRED_PATH.exists():
        with PRED_PATH.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if obj.get("ollama_is_hazard", "").lower() == "yes":
                        preds[int(obj["id"])] = obj

    print(f"Gold hazard indicators: {len(gold_his)}")
    print(f"LLM-confirmed hazards: {len(preds)}")
    print(f"Processing all {len(gold_his)} gold HIs for state recovery...\n")

    structured_his = []
    for i, gold in enumerate(gold_his):
        pid = int(gold['id'])
        hi_id = f"HI-{i+1:03d}"

        # Use gold annotations as the base (more reliable than LLM extraction)
        condition = gold.get('condition', '')
        operation = gold.get('operation', '')
        paragraph = gold.get('paragraph', '')

        print(f"Processing {hi_id} (ID={pid}, section {gold['section_id']})...")

        # Call LLM for detailed recovery
        system, user = build_recovery_prompt(paragraph, condition, operation)
        recovery = call_ollama(system, user)

        structured = {
            "hi_id": hi_id,
            "paragraph_id": pid,
            "source_section": gold.get('section_id', ''),
            "section_title": gold.get('section_title', ''),
            "paragraph_text": paragraph[:300],

            # From gold annotations
            "gold_condition": condition,
            "gold_operation": operation,
            "gold_state": gold.get('state', ''),
            "gold_hazard_type": gold.get('hazard_type', ''),

            # From LLM recovery
            "trigger_message": recovery.get("trigger_message", ""),
            "trigger_direction": recovery.get("trigger_direction", "unknown"),
            "entity": recovery.get("entity", "unknown"),
            "pre_state": recovery.get("pre_state", ""),
            "post_state": recovery.get("post_state", ""),
            "operations": recovery.get("operations", []),
            "affected_contexts": recovery.get("affected_contexts", []),
            "timer_impacts": recovery.get("timer_impacts", []),

            # Whether LLM also detected this HI
            "llm_detected": pid in preds,
        }

        # Validate
        warnings = validate_states(
            structured["pre_state"],
            structured["post_state"],
            structured["timer_impacts"]
        )
        if warnings:
            print(f"  Warnings: {warnings}")

        structured_his.append(structured)
        print(f"  {structured['pre_state'] or '?'} -> {structured['post_state'] or '?'} ({structured['trigger_message'] or '?'})")

        time.sleep(0.1)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for shi in structured_his:
            f.write(json.dumps(shi, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Saved {len(structured_his)} structured HIs to {OUT_PATH}")
    print(f"\nSTATE TRANSITION SUMMARY:")
    print(f"{'='*60}")
    for shi in structured_his:
        detected = "✓" if shi["llm_detected"] else "✗"
        arrow = f"{shi['pre_state'] or '?'} -> {shi['post_state'] or '?'}"
        print(f"  [{detected}] {shi['hi_id']}: {arrow}  (on {shi['trigger_message'] or '?'})")

if __name__ == "__main__":
    main()