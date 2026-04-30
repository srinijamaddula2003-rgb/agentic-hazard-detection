"""
AGENTIC STEP 2b: Controlled Reflexion with Validation
======================================================
Lesson from Step 2: Naive reflexion rescued true hazards but also introduced
22 false positives, destroying precision.

Fix: Only apply reflexion to HIGH-CONFIDENCE candidates (paragraphs with
multiple strong signals), and add a validation pass that challenges
rescued paragraphs to filter out false positives.

Pipeline: Consensus → Selective Reflexion → Validation → Final Decision
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
OUT_PATH = Path("data/processed/reflexion_v2_predictions.jsonl")

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5"

# ── Keyword lists ────────────────────────────────────────────────────────────
STRONG_RISKY = [
    "abort", "discard", "clear", "invalidate", "reset",
    "delete", "wipe", "flush"
]
MEDIUM_RISKY = [
    "release", "deactivate", "deactivation", "detach", "terminate",
    "reject", "remove", "revoke", "cancel", "stop", "drop"
]

# ── Paragraphs that should NEVER be classified as hazards ────────────────────
HARD_NO_PATTERNS = [
    "this message is sent by",
    "the state ", "the substate ",
    "the purpose of the ",
    "editor's note",
    "is defined as", "refers to",
    "for the purposes of",
    "in this specification",
    "abbreviation for",
    "see table", "see subclause",
]


def call_llm(system: str, user: str) -> str:
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
    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()
    return text


def parse_json(text: str) -> dict:
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


def compute_risk_score(paragraph: str) -> tuple:
    """
    Compute a risk score for a paragraph. Higher = more likely a real hazard.
    Returns (score, reasons_list).
    """
    t = paragraph.lower()
    score = 0
    reasons = []

    # Hard block: definitional/structural paragraphs get score 0
    if any(t.startswith(p) for p in HARD_NO_PATTERNS):
        return 0, ["blocked: definitional/structural paragraph"]

    # Strong destructive verbs
    for word in STRONG_RISKY:
        if word in t:
            score += 3
            reasons.append(f"strong verb '{word}'")

    # Medium risky operations
    for word in MEDIUM_RISKY:
        if word in t:
            score += 1
            reasons.append(f"risky op '{word}'")

    # Mandatory actions ("shall" + destructive verb)
    if "shall" in t:
        for v in STRONG_RISKY + MEDIUM_RISKY:
            if v in t:
                score += 2
                reasons.append("mandatory action (shall + risky verb)")
                break

    # Conditional triggers
    conditionals = ["if ", "upon ", "when ", "on receipt", "on reception"]
    if any(c in t for c in conditionals):
        score += 2
        reasons.append("conditional trigger")

    # State transitions
    if ("enter" in t and "state" in t) or ("enter state" in t):
        score += 2
        reasons.append("state transition")

    # Specific high-risk state targets
    if "emm-deregistered" in t and ("enter" in t or "transition" in t or "->" in t):
        score += 3
        reasons.append("transition to EMM-DEREGISTERED")

    if "bearer context inactive" in t and ("enter" in t or "transition" in t):
        score += 3
        reasons.append("transition to BEARER CONTEXT INACTIVE")

    # Timer stops (often indicate state changes)
    if "stop" in t and ("timer" in t or re.search(r't3\d{3}', t)):
        score += 2
        reasons.append("timer stop")

    return score, reasons


def build_reflexion_prompt(paragraph: str, reasons: list):
    system = (
        "You are a strict LTE protocol security analyst. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\"} or {\"is_hazard\":\"no\"}. "
        "No extra text."
    )
    user = f"""
Task: Does this paragraph contain a HAZARD INDICATOR?

A hazard indicator MUST have ALL of these:
1. A SPECIFIC trigger message or event (e.g., "upon receipt of X message")
2. A DESTRUCTIVE action as a result (delete, abort, deactivate, clear, reset, stop timer + state change)
3. The action causes LOSS of context, credentials, connectivity, or security state

The following are NOT hazard indicators:
- Descriptions of what a procedure does ("The detach procedure is used to...")
- Definitions of states ("EMM-DEREGISTERED: no EMM context exists")
- Normal successful procedure completions
- Sending a reject message (the SENDER's action is not hazardous; the RECEIVER's response might be)

Suspicious elements found in this paragraph: {'; '.join(reasons)}

Paragraph:
\"\"\"{paragraph}\"\"\"

Does this paragraph describe a SPECIFIC message/event triggering a DESTRUCTIVE action?
Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def build_validation_prompt(paragraph: str):
    """
    Challenge a rescued paragraph — argue AGAINST it being a hazard.
    Only if it survives this challenge do we accept it.
    """
    system = (
        "You are a skeptical LTE protocol reviewer. Your job is to find reasons "
        "why a paragraph is NOT a hazard indicator. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\"} or {\"is_hazard\":\"no\"}. "
        "No extra text."
    )
    user = f"""
A colleague classified this paragraph as a HAZARD INDICATOR. Challenge their assessment.

A paragraph is NOT a hazard indicator if:
- It only DESCRIBES or DEFINES a procedure (e.g., "The detach procedure is used to...")
- It describes a NORMAL, EXPECTED outcome (successful completion, expected state transition)
- The only "risky" element is SENDING a reject message (sending is not destructive)
- It is an editor's note, definition, abbreviation list, or table header
- It describes the PURPOSE of a procedure without specifying trigger-action pairs

A paragraph IS a hazard indicator ONLY if it contains a CONCRETE TRIGGER -> DESTRUCTIVE ACTION pair.
For example: "Upon receipt of X message, the UE shall delete Y" IS a hazard.
"The detach procedure is used by the UE to inform the network" is NOT a hazard.

Paragraph:
\"\"\"{paragraph}\"\"\"

After careful skeptical analysis, is this TRULY a hazard indicator?
Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def build_extractor_prompt(paragraph: str):
    system = (
        "You are an LTE protocol security analyst. "
        "Extract hazard indicator details. Return ONLY valid JSON. No extra text."
    )
    user = f"""
Extract the hazard indicator from this LTE specification paragraph.

Return a JSON object with:
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
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    fp = sum(1 for t, pr in zip(y_true, y_pred) if t == 0 and pr == 1)
    fn = sum(1 for t, pr in zip(y_true, y_pred) if t == 1 and pr == 0)
    return p, r, f, fp, fn


def main():
    gold = pd.read_csv(GOLD_PATH)

    # Load consensus predictions
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

    # Score ALL consensus-NO paragraphs
    candidates = []
    for _, row in gold.iterrows():
        pid = int(row["id"])
        c = consensus.get(pid, {})
        if c.get("consensus_is_hazard", "no") == "no":
            paragraph = str(row["paragraph"])
            score, reasons = compute_risk_score(paragraph)
            candidates.append((pid, paragraph, score, reasons))

    # Sort by score descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Only reflect on HIGH-SCORE candidates (score >= 5)
    THRESHOLD = 5
    high_risk = [(pid, para, score, reasons) for pid, para, score, reasons in candidates if score >= THRESHOLD]
    low_risk = [(pid, para, score, reasons) for pid, para, score, reasons in candidates if score < THRESHOLD]

    print(f"Consensus said NO for {len(candidates)} paragraphs")
    print(f"Risk scoring: {len(high_risk)} high-risk (score >= {THRESHOLD}), {len(low_risk)} low-risk")
    print(f"Only reflecting on {len(high_risk)} high-risk paragraphs...\n")

    reflexion_results = {}
    rescued = 0
    validated = 0
    rejected_by_validator = 0

    for pid, paragraph, score, reasons in high_risk:
        print(f"ID {pid} (score={score}): Reflecting...")

        # ── Reflexion: improved prompt with risk context ──
        sys_r, usr_r = build_reflexion_prompt(paragraph, reasons)
        reflex_text = call_llm(sys_r, usr_r)
        reflex_json = parse_json(reflex_text)
        reflex_answer = str(reflex_json.get("is_hazard", "no")).lower()

        if reflex_answer != "yes":
            reflexion_results[pid] = {"final": "no", "stage": "reflexion_said_no"}
            print(f"  → Reflexion said NO")
            continue

        # ── Validation: challenge the reflexion answer ──
        sys_v, usr_v = build_validation_prompt(paragraph)
        valid_text = call_llm(sys_v, usr_v)
        valid_json = parse_json(valid_text)
        valid_answer = str(valid_json.get("is_hazard", "no")).lower()

        if valid_answer != "yes":
            reflexion_results[pid] = {"final": "no", "stage": "validator_rejected"}
            rejected_by_validator += 1
            print(f"  → Reflexion YES, but VALIDATOR rejected")
            continue

        # Both reflexion and validator agree: accept as hazard
        reflexion_results[pid] = {"final": "yes", "stage": "validated"}
        rescued += 1
        validated += 1
        print(f"  → RESCUED and VALIDATED!")

        time.sleep(0.1)

    # Mark low-risk candidates as not reflected
    for pid, para, score, reasons in low_risk:
        reflexion_results[pid] = {"final": "no", "stage": "below_threshold"}

    print(f"\nReflexion summary:")
    print(f"  Reflected: {len(high_risk)}")
    print(f"  Rescued + Validated: {validated}")
    print(f"  Rejected by validator: {rejected_by_validator}")

    # Build final predictions
    final_preds = []
    for _, row in gold.iterrows():
        pid = int(row["id"])
        c = consensus.get(pid, {})
        r = reflexion_results.get(pid, {})

        # Start with consensus answer
        is_hazard = c.get("consensus_is_hazard", "no")
        condition = c.get("consensus_condition", "")
        operation = c.get("consensus_operation", "")
        state = c.get("consensus_state", "")
        hazard_type = c.get("consensus_hazard_type", "")
        source = c.get("source_model", "none")

        # Override only if reflexion + validation both say yes
        if r.get("final") == "yes":
            is_hazard = "yes"
            source = "reflexion_validated"

            # Extract details for newly rescued paragraphs
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
            "reflexion_stage": r.get("stage", "not_reflected"),
        }
        final_preds.append(record)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in final_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Full Evaluation ──
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in gold.iterrows()]

    # Load all baselines
    def load_pred_labels(path, prefix):
        preds = {}
        if path.exists():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        preds[int(obj["id"])] = obj
        labels = []
        for _, row in gold.iterrows():
            pid = int(row["id"])
            obj = preds.get(pid, {})
            val = str(obj.get(f"{prefix}_is_hazard", "no")).strip().lower()
            labels.append(1 if val == "yes" else 0)
        return labels

    y_llama = load_pred_labels(Path("data/processed/ollama_predictions.jsonl"), "ollama")
    y_qwen = load_pred_labels(Path("data/processed/qwen_predictions.jsonl"), "qwen")
    y_consensus = [1 if consensus.get(int(row["id"]), {}).get("consensus_is_hazard", "no") == "yes" else 0
                   for _, row in gold.iterrows()]
    y_reflexion_v1 = []
    refv1_path = Path("data/processed/reflexion_predictions.jsonl")
    if refv1_path.exists():
        refv1 = {}
        with refv1_path.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    refv1[int(obj["id"])] = obj
        y_reflexion_v1 = [1 if refv1.get(int(row["id"]), {}).get("final_is_hazard", "no") == "yes" else 0
                          for _, row in gold.iterrows()]

    y_reflexion_v2 = [1 if rec["final_is_hazard"] == "yes" else 0 for rec in final_preds]

    print("\n" + "=" * 80)
    print("FULL PROGRESSION: Vanilla → Consensus → Reflexion v1 → Reflexion v2")
    print("=" * 80)
    print(f"{'Method':<45s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 80)

    methods = [
        ("1. Llama 3.2 3B (vanilla)", y_llama),
        ("2. Qwen 3.5 397B (vanilla)", y_qwen),
        ("3. Consensus: Union (Step 1)", y_consensus),
    ]
    if y_reflexion_v1:
        methods.append(("4. Reflexion naive (Step 2 — broken)", y_reflexion_v1))
    methods.append(("5. Reflexion + Validation (Step 2b)", y_reflexion_v2))

    for name, y_pred in methods:
        p, r, f, fp, fn = evaluate(name, y_true, y_pred)
        print(f"{name:<45s} {p:>8.3f} {r:>8.3f} {f:>8.3f} {fp:>5d} {fn:>5d}")

    print(f"\nSaved final predictions to {OUT_PATH}")

    # Show remaining misses
    print("\n--- Still missed ---")
    for rec in final_preds:
        pid = rec["id"]
        gold_label = gold[gold["id"] == pid].iloc[0]["is_hazard"]
        if str(gold_label).strip().lower() == "yes" and rec["final_is_hazard"] == "no":
            para = rec["paragraph"][:100]
            print(f"  ID {pid}: {para}...")

    print("\n--- False positives (if any) ---")
    for rec in final_preds:
        pid = rec["id"]
        gold_label = gold[gold["id"] == pid].iloc[0]["is_hazard"]
        if str(gold_label).strip().lower() != "yes" and rec["final_is_hazard"] == "yes":
            para = rec["paragraph"][:80]
            print(f"  ID {pid} (source={rec['source']}): {para}...")


if __name__ == "__main__":
    main()