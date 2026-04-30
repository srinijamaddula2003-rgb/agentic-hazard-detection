"""
HYBRID MODE: Guided + Open-Ended + Validation
================================================
Combines the best of both approaches:
- Guided mode: high precision, catches well-defined HIs
- Open-ended mode: high recall, discovers novel threats

Strategy:
1. Start with guided mode results (high precision base)
2. For paragraphs guided said NO, check if open-ended said YES
3. Validate those candidates with a focused prompt that checks
   whether the open-ended threat is genuine or speculative
4. Accept only validated discoveries

This gives us: guided precision + open-ended discovery + validation quality control
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time
import sys

GOLD_PATH = Path("data/processed/annotation_sample.csv")
GUIDED_PATH = Path("data/processed/qwen_nothinking_predictions.jsonl")
OPENENDED_PATH = Path("data/processed/openended_predictions.jsonl")
LLAMA_PATH = Path("data/processed/ollama_predictions.jsonl")
OUT_PATH = Path("data/processed/hybrid_predictions.jsonl")

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5"


def call_llm(system: str, user: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 400,
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(LLAMA_CPP_URL, json=payload, timeout=900)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()

    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()

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


def build_validation_prompt(paragraph, threat_description):
    """
    Validate an open-ended discovery. The key question:
    Does this paragraph describe a CONCRETE, ACTIONABLE security risk,
    or is the threat purely speculative/theoretical?
    """
    system = (
        "You are a senior telecom security reviewer. "
        "Return ONLY JSON: {\"is_valid\":\"yes\",\"reason\":\"...\"} or "
        "{\"is_valid\":\"no\",\"reason\":\"...\"}. No extra text."
    )
    user = f"""
An automated security scanner flagged this paragraph as containing a potential threat:

Paragraph:
\"\"\"{paragraph}\"\"\"

Identified threat:
\"\"\"{threat_description}\"\"\"

Your job: Determine if this is a REAL, ACTIONABLE security risk or a FALSE ALARM.

ACCEPT as valid (return "yes") if:
1. The paragraph describes a CONCRETE action/behavior (not just a definition or overview)
2. The threat involves a specific exploitable mechanism (message forgery, state manipulation,
   timer abuse, missing validation, etc.)
3. An attacker could REALISTICALLY trigger this (not purely theoretical)
4. The consequence is HARMFUL (DoS, information leak, bypass, state corruption)

REJECT as false alarm (return "no") if:
1. The paragraph is just a DEFINITION or PURPOSE DESCRIPTION with no actionable behavior
   Example: "The detach procedure is used to inform the network..."
2. The threat is purely SPECULATIVE — the attacker would need unrealistic capabilities
   Example: Requiring full control of the core network infrastructure
3. The paragraph describes a NORMAL procedure and the "threat" is just that procedures exist
   Example: "UE sends ATTACH REQUEST" → flagged as "attacker can send fake attach" — too generic
4. The threat is about EDITOR'S NOTES or FFS items — these are spec process issues, not vulnerabilities
5. The threat requires capabilities BEYOND what the attacker model allows
   (attacker can send/intercept radio messages but cannot compromise core network nodes)

Return ONLY JSON: {{"is_valid":"yes","reason":"..."}} or {{"is_valid":"no","reason":"..."}}
"""
    return system, user


def main():
    df = pd.read_csv(GOLD_PATH)

    # Load all prediction sets
    guided = {}
    if GUIDED_PATH.exists():
        with GUIDED_PATH.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    guided[int(obj["id"])] = obj

    openended = {}
    if OPENENDED_PATH.exists():
        with OPENENDED_PATH.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    openended[int(obj["id"])] = obj

    llama = {}
    if LLAMA_PATH.exists():
        with LLAMA_PATH.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    llama[int(obj["id"])] = obj

    if not openended:
        print("ERROR: Run openended_discovery.py first!")
        return

    # Step 1: Start with consensus base (Llama + Guided Qwen)
    consensus_yes = set()
    for _, row in df.iterrows():
        pid = int(row["id"])
        l = llama.get(pid, {}).get("ollama_is_hazard", "no").lower() == "yes"
        g = guided.get(pid, {}).get("qwen_nothinking_is_hazard", "no").lower() == "yes"
        if l or g:
            consensus_yes.add(pid)

    print(f"Consensus base: {len(consensus_yes)} hazards identified")

    # Step 2: Find open-ended discoveries NOT in consensus
    candidates = []
    for _, row in df.iterrows():
        pid = int(row["id"])
        if pid not in consensus_yes:
            op = openended.get(pid, {})
            if op.get("openended_is_hazard", "no") == "yes":
                threat = op.get("openended_threat", "")
                candidates.append((pid, str(row["paragraph"]), threat))

    print(f"Open-ended discoveries beyond consensus: {len(candidates)}")
    print(f"Validating each candidate...\n")

    # Step 3: Validate each open-ended discovery
    validated = {}
    stats = {"confirmed": 0, "rejected": 0}

    for pid, paragraph, threat in candidates:
        print(f"ID {pid}: {threat[:60]}...")

        sys_v, usr_v = build_validation_prompt(paragraph, threat)
        result = call_llm(sys_v, usr_v)
        is_valid = str(result.get("is_valid", "no")).lower()
        reason = str(result.get("reason", ""))[:80]

        if is_valid == "yes":
            validated[pid] = {"valid": True, "threat": threat, "reason": reason}
            stats["confirmed"] += 1
            print(f"  CONFIRMED: {reason}")
        else:
            validated[pid] = {"valid": False, "reason": reason}
            stats["rejected"] += 1
            print(f"  REJECTED:  {reason}")

        time.sleep(0.05)

    print(f"\nValidation: {stats['confirmed']} confirmed, {stats['rejected']} rejected")

    # Step 4: Build final predictions
    final_preds = []
    for _, row in df.iterrows():
        pid = int(row["id"])
        paragraph = str(row["paragraph"])

        if pid in consensus_yes:
            # Accepted by consensus
            is_hazard = "yes"
            source = "consensus"
            threat = ""
        elif pid in validated and validated[pid]["valid"]:
            # Discovered by open-ended, validated
            is_hazard = "yes"
            source = "openended_validated"
            threat = validated[pid].get("threat", "")
        else:
            is_hazard = "no"
            source = "none"
            threat = ""

        record = {
            "id": pid,
            "paragraph": paragraph,
            "hybrid_is_hazard": is_hazard,
            "source": source,
            "threat": threat,
        }
        final_preds.append(record)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in final_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Evaluation ──
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in df.iterrows()]

    y_llama = [1 if llama.get(int(row["id"]), {}).get("ollama_is_hazard", "no").lower() == "yes" else 0
               for _, row in df.iterrows()]
    y_guided = [1 if guided.get(int(row["id"]), {}).get("qwen_nothinking_is_hazard", "no").lower() == "yes" else 0
                for _, row in df.iterrows()]
    y_open = [1 if openended.get(int(row["id"]), {}).get("openended_is_hazard", "no") == "yes" else 0
              for _, row in df.iterrows()]
    y_consensus = [1 if int(row["id"]) in consensus_yes else 0
                   for _, row in df.iterrows()]
    y_hybrid = [1 if rec["hybrid_is_hazard"] == "yes" else 0 for rec in final_preds]

    def eval_print(name, yt, yp):
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{name:<50s} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {fp:>5d} {fn:>5d}")

    print(f"\n{'='*85}")
    print("COMPLETE PIPELINE COMPARISON")
    print(f"{'='*85}")
    print(f"{'Method':<50s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 85)
    eval_print("1. Llama 3.2 3B (guided)", y_true, y_llama)
    eval_print("2. Qwen 397B (guided)", y_true, y_guided)
    eval_print("3. Open-ended (no rules)", y_true, y_open)
    eval_print("4. Consensus (Llama + Qwen guided)", y_true, y_consensus)
    eval_print("5. HYBRID (consensus + validated open-ended)", y_true, y_hybrid)
    print("-" * 85)
    print(f"{'Bookworm (reported, approx.)':<50s} {'~0.560':>8s} {'~1.000':>8s} {'~0.720':>8s} {'—':>5s} {'—':>5s}")

    # Detail sections
    print(f"\n--- Still missed (FN) ---")
    fn_count = 0
    for rec in final_preds:
        pid = rec["id"]
        gl = str(df[df["id"] == pid].iloc[0]["is_hazard"]).strip().lower()
        if gl == "yes" and rec["hybrid_is_hazard"] == "no":
            print(f"  ID {pid}: {rec['paragraph'][:100]}...")
            fn_count += 1
    if fn_count == 0:
        print("  None!")

    print(f"\n--- False positives (FP) ---")
    fp_count = 0
    for rec in final_preds:
        pid = rec["id"]
        gl = str(df[df["id"] == pid].iloc[0]["is_hazard"]).strip().lower()
        if gl != "yes" and rec["hybrid_is_hazard"] == "yes":
            print(f"  ID {pid} ({rec['source']}): {rec['paragraph'][:60]}...")
            print(f"    Threat: {rec.get('threat', '')[:60]}")
            fp_count += 1
    if fp_count == 0:
        print("  None!")

    print(f"\n--- Validated novel discoveries (beyond gold standard) ---")
    novel = 0
    for rec in final_preds:
        pid = rec["id"]
        gl = str(df[df["id"] == pid].iloc[0]["is_hazard"]).strip().lower()
        if rec["source"] == "openended_validated":
            label = "MATCHES GOLD" if gl == "yes" else "NOVEL (not in gold)"
            print(f"  ID {pid} [{label}]: {rec.get('threat', '')[:80]}")
            if gl != "yes":
                novel += 1

    if novel > 0:
        print(f"\n  {novel} NOVEL threats discovered beyond gold standard!")
        print(f"  These should be manually reviewed for potential new vulnerabilities.")

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()