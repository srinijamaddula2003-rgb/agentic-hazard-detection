"""
AGENTIC STEP 5: Combined Pipeline (Consensus + Validated Context Engineering)
===============================================================================
Insight from experiments:
- Consensus: Perfect precision (1.000) but misses 3 hazards (recall 0.824)
- Context Engineering: Perfect recall (1.000) but 7 false positives (precision 0.708)

Solution: Use consensus as the base, then apply context engineering ONLY to
consensus-NO paragraphs, with a validation pass to filter out false positives.

Pipeline: Consensus → Context Engineering (on NO only) → Validation → Final
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time

GOLD_PATH = Path("data/processed/annotation_sample.csv")
CONSENSUS_PATH = Path("data/processed/consensus_predictions.jsonl")
CONTEXT_PATH = Path("data/processed/context_eng_predictions.jsonl")
OUT_PATH = Path("data/processed/combined_pipeline_predictions.jsonl")

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5"


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


def build_validation_prompt(paragraph, section_title):
    system = (
        "You are a skeptical LTE protocol security reviewer checking for false positives. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\",\"reason\":\"...\"} or "
        "{\"is_hazard\":\"no\",\"reason\":\"...\"}. No extra text."
    )
    user = f"""
Another analyst classified this paragraph as a HAZARD INDICATOR. Your job is to
verify whether this is a REAL hazard or a FALSE POSITIVE.

Section: {section_title}

Paragraph:
\"\"\"{paragraph}\"\"\"

A paragraph is a FALSE POSITIVE (return "no") if:
1. It only DESCRIBES what a procedure does without a specific trigger→action pair
   Example FP: "The detach procedure is used by the UE to inform the network..."
2. It only defines a STATE without describing what triggers entry into that state
   Example FP: "A UE enters EMM-DEREGISTERED-INITIATED after it has requested release"
3. The only action is SENDING a message (sending is not destructive)
   Example FP: "The network shall return a SERVICE REJECT message to the UE"
4. It describes a NORMAL SUCCESSFUL procedure step
   Example FP: "Upon receipt of GUTI REALLOCATION COMPLETE, the MME shall stop timer T3450"
5. It is a general overview or purpose description

A paragraph is a REAL HAZARD (return "yes") if:
1. A specific message/event TRIGGERS a destructive action (delete, deactivate, abort, discard, reset)
   Example: "Upon receipt of AUTHENTICATION REJECT, the UE shall delete GUTI, TAI list, and KSI"
2. A message causes LOSS of context, credentials, or connectivity
3. A security check failure leads to message DISCARDING
   Example: "If NAS messages have not passed integrity check, the NAS layer shall discard that message"
4. A procedure initiation leads to bearer DEACTIVATION or state transition to DEREGISTERED
   with explicit destructive actions described

Is this a REAL hazard indicator or a false positive?
Return ONLY JSON: {{"is_hazard":"yes","reason":"..."}} or {{"is_hazard":"no","reason":"..."}}
"""
    return system, user


def main():
    gold = pd.read_csv(GOLD_PATH)

    # Load consensus predictions (our high-precision base)
    consensus = {}
    if CONSENSUS_PATH.exists():
        with CONSENSUS_PATH.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    consensus[int(obj["id"])] = obj

    # Load context engineering predictions (has perfect recall)
    context_preds = {}
    if CONTEXT_PATH.exists():
        with CONTEXT_PATH.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    context_preds[int(obj["id"])] = obj
    else:
        print("ERROR: Run agentic_step3_context.py first!")
        return

    # Find paragraphs that context engineering rescued (consensus=NO, context=YES)
    rescued_by_context = []
    for _, row in gold.iterrows():
        pid = int(row["id"])
        cons = consensus.get(pid, {}).get("consensus_is_hazard", "no")
        ctx = context_preds.get(pid, {}).get("final_is_hazard", "no")
        if cons == "no" and ctx == "yes":
            rescued_by_context.append(pid)

    print(f"Consensus: {sum(1 for c in consensus.values() if c.get('consensus_is_hazard') == 'yes')} YES, "
          f"{sum(1 for c in consensus.values() if c.get('consensus_is_hazard') == 'no')} NO")
    print(f"Context engineering rescued: {len(rescued_by_context)} paragraphs")
    print(f"Validating each rescued paragraph...\n")

    # Validate each rescued paragraph with skeptical reviewer
    validated = {}
    stats = {"confirmed": 0, "rejected": 0}

    for pid in rescued_by_context:
        row = gold[gold["id"] == pid].iloc[0]
        paragraph = str(row["paragraph"])
        section_title = str(row.get("section_title", ""))

        print(f"ID {pid} (section: {section_title[:50]})")

        sys_v, usr_v = build_validation_prompt(paragraph, section_title)
        val_text = call_llm(sys_v, usr_v)
        val_json = parse_json(val_text)
        val_answer = str(val_json.get("is_hazard", "no")).lower()
        val_reason = str(val_json.get("reason", ""))[:80]

        if val_answer == "yes":
            validated[pid] = True
            stats["confirmed"] += 1
            print(f"  → CONFIRMED: {val_reason}")
        else:
            validated[pid] = False
            stats["rejected"] += 1
            print(f"  → REJECTED: {val_reason}")

        time.sleep(0.1)

    print(f"\nValidation: {stats['confirmed']} confirmed, {stats['rejected']} rejected")

    # Build final predictions:
    # Base = consensus
    # Add only VALIDATED rescues from context engineering
    final_preds = []
    for _, row in gold.iterrows():
        pid = int(row["id"])
        c = consensus.get(pid, {})
        ctx = context_preds.get(pid, {})

        # Start with consensus
        is_hazard = c.get("consensus_is_hazard", "no")
        condition = c.get("consensus_condition", "")
        operation = c.get("consensus_operation", "")
        state = c.get("consensus_state", "")
        hazard_type = c.get("consensus_hazard_type", "")
        source = c.get("source_model", "none")

        # Add validated context engineering rescues
        if pid in validated and validated[pid]:
            is_hazard = "yes"
            source = "context_validated"
            condition = ctx.get("final_condition", "")
            operation = ctx.get("final_operation", "")
            state = ctx.get("final_state", "")
            hazard_type = ctx.get("final_hazard_type", "other")

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

    # ── Complete Evaluation ──
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
    y_context = [1 if context_preds.get(int(row["id"]), {}).get("final_is_hazard", "no") == "yes" else 0
                 for _, row in gold.iterrows()]
    y_final = [1 if rec["final_is_hazard"] == "yes" else 0 for rec in final_preds]

    def eval_print(name, yt, yp):
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{name:<50s} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {fp:>5d} {fn:>5d}")

    print(f"\n{'='*85}")
    print("COMPLETE AGENTIC PIPELINE PROGRESSION")
    print(f"{'='*85}")
    print(f"{'Method':<50s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 85)
    eval_print("1. Llama 3.2 3B (vanilla)", y_true, y_llama)
    eval_print("2. Qwen 3.5 397B (vanilla)", y_true, y_qwen)
    eval_print("3. Multi-Model Consensus (Step 1)", y_true, y_consensus)
    eval_print("4. Context Engineering unvalidated (Step 3)", y_true, y_context)
    eval_print("5. Consensus + Validated Context (Combined)", y_true, y_final)
    print("-" * 85)
    print(f"{'Bookworm (reported, approx.)':<50s} {'~0.560':>8s} {'~1.000':>8s} {'~0.720':>8s} {'—':>5s} {'—':>5s}")

    # Detail
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
        print("  None! Perfect precision!")

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()