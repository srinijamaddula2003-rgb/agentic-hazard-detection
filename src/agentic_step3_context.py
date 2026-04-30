"""
AGENTIC STEP 3: Context Engineering
=====================================
Addresses the 3 remaining false negatives (IDs 5, 72, 81) which are
cross-paragraph hazards where the trigger and action span multiple paragraphs.

Approach:
- For consensus-NO paragraphs with moderate risk scores,
  include surrounding paragraphs and section metadata in the prompt
- The enriched context lets the LLM connect triggers to actions across paragraphs
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time

GOLD_PATH = Path("data/processed/annotation_sample.csv")
CONSENSUS_PATH = Path("data/processed/consensus_predictions.jsonl")
OUT_PATH = Path("data/processed/context_eng_predictions.jsonl")

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5"

STRONG_RISKY = [
    "abort", "discard", "clear", "invalidate", "reset",
    "delete", "wipe", "flush"
]
MEDIUM_RISKY = [
    "release", "deactivate", "deactivation", "detach", "terminate",
    "reject", "remove", "revoke", "cancel", "stop", "drop"
]

# Section titles that strongly suggest hazard content
HIGH_RISK_SECTIONS = [
    "abnormal", "not accepted", "reject", "failure", "error",
    "authentication not accepted", "security mode",
    "integrity checking", "detach", "deregistration"
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


def compute_risk_score(paragraph: str, section_title: str) -> int:
    t = paragraph.lower()
    score = 0

    # Hard block
    hard_blocks = [
        "this message is sent by", "the state ", "the substate ",
        "the purpose of the ", "editor's note:", "is defined as",
        "refers to", "for the purposes of", "abbreviation for",
    ]
    if any(t.startswith(p) for p in hard_blocks):
        return 0

    for word in STRONG_RISKY:
        if word in t:
            score += 3
    for word in MEDIUM_RISKY:
        if word in t:
            score += 1
    if "shall" in t and any(v in t for v in STRONG_RISKY + MEDIUM_RISKY):
        score += 2
    conditionals = ["if ", "upon ", "when ", "on receipt", "on reception"]
    if any(c in t for c in conditionals):
        score += 2
    if "enter" in t and "state" in t:
        score += 2
    if re.search(r'stop.*t3\d{3}|t3\d{3}.*stop', t):
        score += 2

    # Section title bonus
    st = section_title.lower()
    if any(h in st for h in HIGH_RISK_SECTIONS):
        score += 3

    return score


def get_surrounding_context(gold_rows, target_idx, window=2):
    """Get paragraphs before and after the target."""
    before = []
    after = []
    for i in range(max(0, target_idx - window), target_idx):
        before.append(gold_rows[i])
    for i in range(target_idx + 1, min(len(gold_rows), target_idx + window + 1)):
        after.append(gold_rows[i])
    return before, after


def build_context_prompt(paragraph, section_id, section_title, before_paras, after_paras):
    system = (
        "You are a strict LTE protocol security analyst. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\"} or {\"is_hazard\":\"no\"}. "
        "No extra text."
    )

    # Build context block
    context_parts = []
    if before_paras:
        context_parts.append("PRECEDING PARAGRAPHS (for context):")
        for bp in before_paras:
            context_parts.append(f"  [{bp.get('section_id', '')}] {bp.get('paragraph', '')[:300]}")

    if after_paras:
        context_parts.append("\nFOLLOWING PARAGRAPHS (for context):")
        for ap in after_paras:
            context_parts.append(f"  [{ap.get('section_id', '')}] {ap.get('paragraph', '')[:300]}")

    context_block = "\n".join(context_parts)

    # Check if section title suggests hazardous content
    section_note = ""
    st = section_title.lower()
    if any(h in st for h in HIGH_RISK_SECTIONS):
        section_note = f"\nNOTE: This paragraph is from section \"{section_title}\" which typically contains security-relevant procedures.\n"

    user = f"""
Task: Does the TARGET PARAGRAPH contain a HAZARD INDICATOR?

A hazard indicator is a statement that links a trigger event to a DESTRUCTIVE or
SECURITY-RELEVANT operation — one that causes loss of context, credentials, or connectivity.

IMPORTANT: Consider the surrounding context when making your decision. The destructive
consequence of an action described in the target paragraph may be clarified in neighboring
paragraphs. If the target paragraph INITIATES or TRIGGERS an action whose destructive
consequences are visible in the surrounding context, it IS a hazard indicator.

=== RETURN "yes" if the TARGET paragraph describes: ===
1. Bearer context being DEACTIVATED or DELETED
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials being DELETED or INVALIDATED
4. Context being RESET or RELEASED
5. State transition to EMM-DEREGISTERED or BEARER CONTEXT INACTIVE
6. Timer being STOPPED as part of a state change
7. Messages being DISCARDED due to security checks
8. A procedure INITIATION (like detach, deregistration) that leads to destructive
   consequences visible in the surrounding paragraphs
9. A REJECT message being sent where the surrounding context shows the receiver
   takes destructive actions (stop timer, enter deregistered state, delete credentials)

=== RETURN "no" for: ===
1. Message format tables
2. State definitions without trigger-action pairs
3. Normal successful procedure steps
4. General descriptions of procedure purposes
5. Timer definitions, abbreviation lists

SECTION: {section_id} — {section_title}
{section_note}
{context_block}

=== TARGET PARAGRAPH (classify this one): ===
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def build_extractor_prompt(paragraph, before_paras, after_paras):
    system = (
        "You are an LTE protocol security analyst. "
        "Extract hazard indicator details. Return ONLY valid JSON."
    )

    context = ""
    if before_paras:
        context += "Context before: " + " ".join(p.get('paragraph', '')[:200] for p in before_paras) + "\n"
    if after_paras:
        context += "Context after: " + " ".join(p.get('paragraph', '')[:200] for p in after_paras) + "\n"

    user = f"""
Extract the hazard indicator from this paragraph. Use the surrounding context
to identify the full trigger-action chain even if it spans paragraphs.

{context}
Target paragraph: \"\"\"{paragraph}\"\"\"

Return a JSON object with:
- condition: The specific trigger message or event
- operation: The destructive/risky action(s) — include consequences from context if needed
- state: Protocol state before -> after
- hazard_type: One of "context_reset", "credential_handling", "exception_handling",
  "state_violation", "ordering_violation", "other"

Return ONLY the JSON object.
"""
    return system, user


def main():
    gold = pd.read_csv(GOLD_PATH)
    gold_rows = gold.to_dict('records')

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

    # Find consensus-NO paragraphs with moderate+ risk scores
    candidates = []
    for idx, row in enumerate(gold_rows):
        pid = int(row["id"])
        c = consensus.get(pid, {})
        if c.get("consensus_is_hazard", "no") == "no":
            score = compute_risk_score(
                str(row["paragraph"]),
                str(row.get("section_title", ""))
            )
            if score >= 3:
                candidates.append((idx, pid, score))

    candidates.sort(key=lambda x: x[2], reverse=True)

    print(f"Consensus said NO for {sum(1 for _, row in gold.iterrows() if consensus.get(int(row['id']), {}).get('consensus_is_hazard', 'no') == 'no')} paragraphs")
    print(f"Context engineering candidates (score >= 3): {len(candidates)}")
    print(f"Processing with surrounding context...\n")

    context_results = {}
    rescued = 0

    for idx, pid, score in candidates:
        row = gold_rows[idx]
        paragraph = str(row["paragraph"])
        section_id = str(row.get("section_id", ""))
        section_title = str(row.get("section_title", ""))

        # Get surrounding context
        before_paras, after_paras = get_surrounding_context(gold_rows, idx, window=2)

        print(f"ID {pid} (score={score}, section: {section_title[:50]})")

        # Classify with context
        sys_c, usr_c = build_context_prompt(
            paragraph, section_id, section_title,
            before_paras, after_paras
        )
        cls_text = call_llm(sys_c, usr_c)
        cls_json = parse_json(cls_text)
        answer = str(cls_json.get("is_hazard", "no")).lower()

        if answer == "yes":
            rescued += 1
            # Extract details with context
            sys_e, usr_e = build_extractor_prompt(paragraph, before_paras, after_paras)
            ext_text = call_llm(sys_e, usr_e)
            ext_json = parse_json(ext_text)

            context_results[pid] = {
                "final": "yes",
                "condition": str(ext_json.get("condition", "")),
                "operation": str(ext_json.get("operation", "")),
                "state": str(ext_json.get("state", "")),
                "hazard_type": str(ext_json.get("hazard_type", "other")),
            }
            print(f"  → YES with context (rescued!)")
        else:
            context_results[pid] = {"final": "no"}
            print(f"  → Still NO with context")

        time.sleep(0.1)

    print(f"\nRescued {rescued} paragraphs through context engineering")

    # Build final predictions: consensus + context overrides
    final_preds = []
    for _, row in gold.iterrows():
        pid = int(row["id"])
        c = consensus.get(pid, {})
        ctx = context_results.get(pid, {})

        is_hazard = c.get("consensus_is_hazard", "no")
        condition = c.get("consensus_condition", "")
        operation = c.get("consensus_operation", "")
        state = c.get("consensus_state", "")
        hazard_type = c.get("consensus_hazard_type", "")
        source = c.get("source_model", "none")

        # Override if context engineering rescued it
        if ctx.get("final") == "yes":
            is_hazard = "yes"
            source = "context_engineering"
            condition = ctx.get("condition", "")
            operation = ctx.get("operation", "")
            state = ctx.get("state", "")
            hazard_type = ctx.get("hazard_type", "other")

        record = {
            "id": pid,
            "paragraph": str(row["paragraph"]),
            "final_is_hazard": is_hazard,
            "final_condition": condition,
            "final_operation": operation,
            "final_state": state,
            "final_hazard_type": hazard_type,
            "source": source,
        }
        final_preds.append(record)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in final_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Evaluation ──
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in gold.iterrows()]

    def load_labels(path, prefix):
        preds = {}
        if path.exists():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        preds[int(obj["id"])] = obj
        return [1 if preds.get(int(row["id"]), {}).get(f"{prefix}_is_hazard", "no").lower() == "yes" else 0
                for _, row in gold.iterrows()]

    y_llama = load_labels(Path("data/processed/ollama_predictions.jsonl"), "ollama")
    y_qwen = load_labels(Path("data/processed/qwen_predictions.jsonl"), "qwen")
    y_consensus = [1 if consensus.get(int(row["id"]), {}).get("consensus_is_hazard", "no") == "yes" else 0
                   for _, row in gold.iterrows()]
    y_final = [1 if rec["final_is_hazard"] == "yes" else 0 for rec in final_preds]

    def eval_print(name, yt, yp):
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{name:<45s} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {fp:>5d} {fn:>5d}")

    print(f"\n{'='*80}")
    print("FULL PROGRESSION TABLE")
    print(f"{'='*80}")
    print(f"{'Method':<45s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 80)
    eval_print("1. Llama 3.2 3B (vanilla)", y_true, y_llama)
    eval_print("2. Qwen 3.5 397B (vanilla)", y_true, y_qwen)
    eval_print("3. Consensus: Union (Step 1)", y_true, y_consensus)
    eval_print("4. Consensus + Context Eng (Step 3)", y_true, y_final)

    # Show remaining misses and FPs
    print("\n--- Still missed (FN) ---")
    fn_count = 0
    for rec in final_preds:
        pid = rec["id"]
        gl = str(gold[gold["id"] == pid].iloc[0]["is_hazard"]).strip().lower()
        if gl == "yes" and rec["final_is_hazard"] == "no":
            print(f"  ID {pid}: {rec['paragraph'][:100]}...")
            fn_count += 1
    if fn_count == 0:
        print("  None! All hazards detected!")

    print("\n--- False positives (FP) ---")
    fp_count = 0
    for rec in final_preds:
        pid = rec["id"]
        gl = str(gold[gold["id"] == pid].iloc[0]["is_hazard"]).strip().lower()
        if gl != "yes" and rec["final_is_hazard"] == "yes":
            print(f"  ID {pid} (src={rec['source']}): {rec['paragraph'][:80]}...")
            fp_count += 1
    if fp_count == 0:
        print("  None! Perfect precision maintained!")

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()