"""
AGENTIC STEP 2: Reflexion (Self-Correction)
=============================================
Takes paragraphs where BOTH models said "no" but we suspect might be hazards,
and runs them through a reflection loop:
  1. First pass: classify as before
  2. Reflection: point out specific risky elements and ask model to reconsider
  3. Final decision based on reflected answer

Uses Qwen 3.5 397B via llama.cpp for reflection (stronger reasoning helps here).
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time
from sklearn.metrics import precision_score, recall_score, f1_score

GOLD_PATH = Path("data/processed/annotation_sample.csv")
CONSENSUS_PATH = Path("data/processed/consensus_predictions.jsonl")
OUT_PATH = Path("data/processed/reflexion_predictions.jsonl")

# llama.cpp endpoint (Qwen via SSH tunnel)
LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5"

# ── Keyword lists (same as original pipeline) ────────────────────────────────
STRONG_RISKY = [
    "abort", "discard", "clear", "invalidate", "reset",
    "delete", "wipe", "flush"
]
MEDIUM_RISKY = [
    "release", "deactivate", "deactivation", "detach", "terminate",
    "reject", "remove", "revoke", "cancel", "stop", "drop"
]
STATE_KEYWORDS = [
    "emm-deregistered", "emm-registered", "bearer context inactive",
    "bearer context active", "emm-null", "emm-connected", "emm-idle"
]
TIMER_KEYWORDS = [
    "t3410", "t3412", "t3417", "t3418", "t3421", "t3422",
    "t3430", "t3450", "t3460", "t3470"
]


def call_llm(system: str, user: str) -> str:
    """Call Qwen via llama.cpp and return raw text response."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 500,
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(LLAMA_CPP_URL, json=payload, timeout=900)
    r.raise_for_status()

    text = r.json()["choices"][0]["message"]["content"].strip()

    # Strip <think>...</think> tags from Qwen's reasoning mode
    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()

    return text


def parse_json(text: str) -> dict:
    """Try to extract JSON from LLM response."""
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return {}
        return {}


def extract_risky_elements(paragraph: str) -> str:
    """Extract specific risky elements from a paragraph for the reflection prompt."""
    t = paragraph.lower()
    found = []

    for word in STRONG_RISKY:
        if word in t:
            found.append(f"destructive verb '{word}'")
    for word in MEDIUM_RISKY:
        if word in t:
            found.append(f"risky operation '{word}'")
    for state in STATE_KEYWORDS:
        if state in t:
            found.append(f"state reference '{state.upper()}'")
    for timer in TIMER_KEYWORDS:
        if timer in t:
            found.append(f"timer '{timer.upper()}'")

    # Check for state transitions (→ pattern in text)
    if "enter" in t and "state" in t:
        found.append("state transition")
    if "shall" in t and any(v in t for v in ["delete", "discard", "abort", "release", "deactivate", "stop"]):
        found.append("mandatory destructive action ('shall' + destructive verb)")

    return "; ".join(found) if found else "no obvious risky elements"


def build_first_pass_prompt(paragraph: str):
    system = (
        "You are a strict LTE protocol security analyst. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\"} or {\"is_hazard\":\"no\"}. "
        "No extra text."
    )
    user = f"""
Task: Does this LTE specification paragraph contain a HAZARD INDICATOR?

A hazard indicator is a statement that links a trigger event to a DESTRUCTIVE or
SECURITY-RELEVANT operation — specifically one that causes loss of context, credentials,
or connectivity.

Return "yes" if the paragraph describes:
1. Bearer context being DEACTIVATED or DELETED
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials (GUTI, KSI, TAI list) being DELETED or INVALIDATED
4. Context being RESET or RELEASED
5. State transition to EMM-DEREGISTERED or BEARER CONTEXT INACTIVE
6. Timer being STOPPED due to an incoming message (timer stop = state change trigger)

Paragraph:
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def build_reflection_prompt(paragraph: str, first_answer: str, risky_elements: str):
    system = (
        "You are a strict LTE protocol security analyst performing a second review. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\",\"reason\":\"...\"} or "
        "{\"is_hazard\":\"no\",\"reason\":\"...\"}. No extra text."
    )
    user = f"""
You previously classified this paragraph as: {first_answer}

Now RECONSIDER carefully. Here are specific elements I found in this paragraph:
{risky_elements}

The key question is: Does this paragraph describe a SPECIFIC MESSAGE or EVENT that
triggers a DESTRUCTIVE or STATE-CHANGING operation?

Remember:
- "stop timer" + "enter state X" = hazard (timer stop triggered by a message IS a state change)
- A message causing state transition to DEREGISTERED = hazard (loss of context)
- Procedure being "extended" to allow rejection = hazard (rejection path exists)
- Credential deletion or invalidation = hazard
- Sending a reject message alone is NOT a hazard (but the RECEIVER's response might be)

Paragraph:
\"\"\"{paragraph}\"\"\"

Think step by step:
1. What specific message or event is described?
2. What operation happens as a result?
3. Is the operation destructive (deletes, stops, aborts, deactivates, resets)?
4. Could an attacker trigger this deliberately?

Return ONLY JSON: {{"is_hazard":"yes","reason":"..."}} or {{"is_hazard":"no","reason":"..."}}
"""
    return system, user


def build_extractor_prompt(paragraph: str):
    """Same extractor prompt as original pipeline for consistency."""
    system = (
        "You are an LTE protocol security analyst. "
        "Extract hazard indicator details. Return ONLY valid JSON. No extra text."
    )
    user = f"""
Extract the hazard indicator from this LTE specification paragraph.

Return a JSON object with these fields:
- condition: The specific trigger message or event
- operation: The destructive/risky action(s)
- state: Protocol state before -> after
- hazard_type: One of "context_reset", "credential_handling", "exception_handling",
  "state_violation", "ordering_violation", "other"

Paragraph:
\"\"\"{paragraph}\"\"\"

Return ONLY the JSON object.
"""
    return system, user


def evaluate(name, y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    fp = sum(1 for t, pr in zip(y_true, y_pred) if t == 0 and pr == 1)
    fn = sum(1 for t, pr in zip(y_true, y_pred) if t == 1 and pr == 0)
    return p, r, f, fp, fn


def main():
    gold = pd.read_csv(GOLD_PATH)

    # Load consensus predictions as base
    consensus = {}
    if CONSENSUS_PATH.exists():
        with CONSENSUS_PATH.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    consensus[int(obj["id"])] = obj
    else:
        print("ERROR: Run agentic_step1_consensus.py first!")
        return

    # Identify paragraphs where consensus said NO
    # These are candidates for reflexion
    consensus_no_ids = []
    for _, row in gold.iterrows():
        pid = int(row["id"])
        c = consensus.get(pid, {})
        if c.get("consensus_is_hazard", "no") == "no":
            consensus_no_ids.append(pid)

    print(f"Consensus said NO for {len(consensus_no_ids)} paragraphs")
    print(f"Running reflexion on these paragraphs...\n")

    # Run reflexion on consensus-NO paragraphs
    reflexion_results = {}
    rescued = 0

    for pid in consensus_no_ids:
        row = gold[gold["id"] == pid].iloc[0]
        paragraph = str(row["paragraph"])

        # Check if there are risky elements worth reflecting on
        risky_elements = extract_risky_elements(paragraph)
        if risky_elements == "no obvious risky elements":
            # Skip paragraphs with nothing to reflect on
            reflexion_results[pid] = {
                "reflexion_is_hazard": "no",
                "reflexion_reason": "no risky elements found",
                "was_reflected": False,
            }
            continue

        print(f"ID {pid}: Reflecting (found: {risky_elements[:80]})")

        # First pass
        sys1, usr1 = build_first_pass_prompt(paragraph)
        first_text = call_llm(sys1, usr1)
        first_json = parse_json(first_text)
        first_answer = str(first_json.get("is_hazard", "no")).lower()

        if first_answer == "yes":
            # Model already says yes on improved prompt — accept it
            reflexion_results[pid] = {
                "reflexion_is_hazard": "yes",
                "reflexion_reason": "accepted on improved first pass",
                "was_reflected": False,
            }
            rescued += 1
            print(f"  → YES on first pass (rescued!)")
        else:
            # Reflection step
            sys2, usr2 = build_reflection_prompt(paragraph, first_answer, risky_elements)
            reflect_text = call_llm(sys2, usr2)
            reflect_json = parse_json(reflect_text)
            reflected_answer = str(reflect_json.get("is_hazard", "no")).lower()
            reason = str(reflect_json.get("reason", ""))

            reflexion_results[pid] = {
                "reflexion_is_hazard": reflected_answer,
                "reflexion_reason": reason[:200],
                "was_reflected": True,
            }

            if reflected_answer == "yes":
                rescued += 1
                print(f"  → YES after reflection (rescued!): {reason[:60]}")
            else:
                print(f"  → Still NO after reflection: {reason[:60]}")

        time.sleep(0.1)

    print(f"\nRescued {rescued} paragraphs through reflexion")

    # Build final predictions: consensus + reflexion overrides
    final_preds = []
    for _, row in gold.iterrows():
        pid = int(row["id"])
        c = consensus.get(pid, {})
        r = reflexion_results.get(pid, {})

        # Start with consensus answer
        is_hazard = c.get("consensus_is_hazard", "no")
        source = c.get("source_model", "none")
        condition = c.get("consensus_condition", "")
        operation = c.get("consensus_operation", "")
        state = c.get("consensus_state", "")
        hazard_type = c.get("consensus_hazard_type", "")

        # Override if reflexion said yes
        if r.get("reflexion_is_hazard") == "yes":
            is_hazard = "yes"
            source = "reflexion"

            # Run extractor for newly rescued paragraphs
            if condition == "":
                paragraph = str(row["paragraph"])
                sys_ext, usr_ext = build_extractor_prompt(paragraph)
                ext_text = call_llm(sys_ext, usr_ext)
                ext_json = parse_json(ext_text)
                condition = str(ext_json.get("condition", ""))
                operation = str(ext_json.get("operation", ""))
                state = str(ext_json.get("state", ""))
                hazard_type = str(ext_json.get("hazard_type", "other"))

        record = {
            "id": pid,
            "paragraph": str(row["paragraph"]),
            "final_is_hazard": is_hazard,
            "final_condition": condition,
            "final_operation": operation,
            "final_state": state,
            "final_hazard_type": hazard_type,
            "source": source,
            "was_reflected": r.get("was_reflected", False),
            "reflexion_reason": r.get("reflexion_reason", ""),
        }
        final_preds.append(record)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in final_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Evaluate all methods
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in gold.iterrows()]

    # Load individual baselines
    llama_preds = {}
    if Path("data/processed/ollama_predictions.jsonl").exists():
        with open("data/processed/ollama_predictions.jsonl") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    llama_preds[int(obj["id"])] = obj

    qwen_preds = {}
    if Path("data/processed/qwen_predictions.jsonl").exists():
        with open("data/processed/qwen_predictions.jsonl") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    qwen_preds[int(obj["id"])] = obj

    y_llama = [1 if llama_preds.get(int(row["id"]), {}).get("ollama_is_hazard", "no").lower() == "yes" else 0
               for _, row in gold.iterrows()]
    y_qwen = [1 if qwen_preds.get(int(row["id"]), {}).get("qwen_is_hazard", "no").lower() == "yes" else 0
              for _, row in gold.iterrows()]
    y_consensus = [1 if consensus.get(int(row["id"]), {}).get("consensus_is_hazard", "no") == "yes" else 0
                   for _, row in gold.iterrows()]
    y_reflexion = [1 if rec["final_is_hazard"] == "yes" else 0 for rec in final_preds]

    print("\n" + "=" * 75)
    print("PROGRESSION: Vanilla → Consensus → Reflexion")
    print("=" * 75)
    print(f"{'Method':<40s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'FP':>5s} {'FN':>5s}")
    print("-" * 75)

    for name, y_pred in [
        ("1. Llama 3.2 3B (vanilla)", y_llama),
        ("2. Qwen 3.5 397B (vanilla)", y_qwen),
        ("3. Consensus: Union (Step 1)", y_consensus),
        ("4. Consensus + Reflexion (Step 2)", y_reflexion),
    ]:
        p, r, f, fp, fn = evaluate(name, y_true, y_pred)
        print(f"{name:<40s} {p:>10.3f} {r:>10.3f} {f:>10.3f} {fp:>5d} {fn:>5d}")

    print(f"\nSaved final predictions to {OUT_PATH}")

    # Show what's still missed
    print("\n--- Still missed after Reflexion ---")
    for rec in final_preds:
        pid = rec["id"]
        gold_label = gold[gold["id"] == pid].iloc[0]["is_hazard"]
        if str(gold_label).strip().lower() == "yes" and rec["final_is_hazard"] == "no":
            para = rec["paragraph"][:100]
            print(f"  ID {pid}: {para}...")
    print("\nThese remaining FNs need Context Engineering (Step 3)")


if __name__ == "__main__":
    main()