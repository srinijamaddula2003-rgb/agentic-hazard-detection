"""
PHASE B2: Controlled Hybrid Discovery
=======================================
The middle ground between fully guided and fully open-ended.

Strategy: Give the LLM the same rules as guided mode, BUT also explicitly
tell it to look BEYOND those rules for threats the rules don't cover.

This answers: "Can LLMs discover things we don't specify as rules?"
while maintaining the precision anchor of guided mode.

Comparison:
  Guided only:     "Here are rules. Follow them strictly."
  Open-ended only: "Find anything dangerous. No rules."
  Hybrid (THIS):   "Here are rules. Follow them. ALSO look for threats
                    the rules don't cover, like DoS, privacy leaks,
                    race conditions, and specification ambiguities."
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time
import sys

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OUT_PATH = Path("data/processed/hybrid_controlled_predictions.jsonl")

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5"

# Same prefilter but RELAXED — we want more candidates to reach the LLM
STRONG_RISKY = [
    "abort", "discard", "clear", "invalidate", "reset",
    "delete", "wipe", "flush"
]
MEDIUM_RISKY = [
    "release", "deactivate", "deactivation", "detach", "terminate",
    "reject", "remove", "revoke", "cancel", "stop", "drop"
]
CONDITION_TRIGGERS = [
    "if ", "unless ", "when ", "before ", "upon ", "on receipt",
    "on reception", "after receiving", "in case", "in the event",
    "initiates", "is initiated"
]
ABNORMAL_TRIGGERS = [
    "cannot", "failed", "failure", "not accepted", "invalid", "mismatch",
    "reject", "rejected", "timer expiry", "timeout", "error", "abnormal",
    "no response", "not available", "missing", "unexpected"
]

# Additional triggers for broader coverage
BROADER_TRIGGERS = [
    "shall send", "shall include", "shall initiate", "shall start",
    "imsi", "guti", "supi", "tmsi", "identity",
    "without", "not required", "not mandatory", "optional",
    "any time", "at any time",
]


def prefilter_broad(paragraph: str) -> bool:
    """Broader prefilter — lets more candidates through to the LLM."""
    t = paragraph.lower()

    # Hard blocks (these are never hazards regardless)
    if t.startswith("this message is sent by"):
        return False
    if any(t.startswith(s) for s in ["the state ", "the substate "]):
        if not any(v in t for v in STRONG_RISKY + MEDIUM_RISKY):
            return False
    if any(b in t for b in ["is defined as", "refers to", "abbreviation for"]):
        if not any(k in t for k in STRONG_RISKY + MEDIUM_RISKY):
            return False

    # Original triggers
    has_cond = any(k in t for k in CONDITION_TRIGGERS)
    has_abnormal = any(k in t for k in ABNORMAL_TRIGGERS)
    has_strong = any(k in t for k in STRONG_RISKY)
    has_medium = any(k in t for k in MEDIUM_RISKY)
    has_broader = any(k in t for k in BROADER_TRIGGERS)

    if (has_cond or has_abnormal) and (has_strong or has_medium):
        return True
    if has_strong:
        return True
    # New: also pass through if paragraph has broader security indicators
    if has_broader and (has_cond or has_medium):
        return True

    return False


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


def build_hybrid_prompt(paragraph):
    """
    Hybrid prompt: guided rules + explicit instruction to discover beyond rules.
    """
    system = (
        "You are a strict LTE protocol security analyst AND an adversarial researcher. "
        "Return ONLY JSON. No extra text."
    )
    user = f"""
Task: Analyze this LTE specification paragraph for security hazards.

=== CATEGORY 1: HAZARD INDICATORS (well-defined) ===
Return "yes" if the paragraph describes:
1. Bearer context being DEACTIVATED or DELETED in response to a message/event
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials (GUTI, KSI, TAI list) being DELETED or INVALIDATED
4. Context being RESET or RELEASED due to a reject or failure
5. State transition to EMM-DEREGISTERED triggered by a message
6. Messages being DISCARDED due to failed security checks

=== CATEGORY 2: BROADER THREATS (beyond standard HI rules) ===
ALSO return "yes" if you identify:
7. Privacy exposure: UE identity (IMSI, IMEI) sent in plaintext or without protection
8. Denial of Service: procedure that an attacker could flood/abuse to exhaust resources
9. Missing validation: UE accepts commands without verifying sender authenticity
10. Race condition: timing between procedures that could be exploited
11. Specification gap: behavior explicitly marked as undefined or "FFS" that creates risk
12. Spoofing opportunity: messages accepted without integrity protection

=== RETURN "no" for: ===
1. Message format tables ("This message is sent by...")
2. Pure state definitions with no actionable behavior
3. General descriptions of procedure purposes
4. Timer/abbreviation/scope definitions
5. Editor's notes that don't describe concrete behaviors

=== Examples: ===

HAZARD - Category 1 (yes):
"Upon receipt of an AUTHENTICATION REJECT message, the mobile station shall set the
update status to EU3, delete the stored GUTI, TAI list, and KSI."
→ Credential deletion triggered by a specific message

HAZARD - Category 2 (yes):
"In state EMM-DEREGISTERED, the UE initiates the attach procedure by sending an
ATTACH REQUEST message to the MME. If there is no valid GUTI available, the UE
shall include the IMSI in the ATTACH REQUEST message."
→ IMSI sent in plaintext when GUTI unavailable — privacy exposure

NOT A HAZARD (no):
"The detach procedure is used by the UE to inform the network that it does not want
to access the EPS any longer."
→ General purpose description, no actionable behavior

Paragraph to classify:
\"\"\"{paragraph}\"\"\"

Return ONLY:
{{"is_hazard":"yes","category":"1 or 2","threat":"brief description"}}
or
{{"is_hazard":"no"}}
"""
    return system, user


def load_done_ids(path: Path):
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("id") is not None:
                        done.add(int(obj["id"]))
                except Exception:
                    continue
    return done


def run(df):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(OUT_PATH)

    if len(done_ids) >= len(df):
        print("Already complete. Run with 'eval' to see results.")
        return

    print(f"Running controlled hybrid discovery...")
    print(f"Already done: {len(done_ids)}, remaining: {len(df) - len(done_ids)}\n")

    stats = {"prefilter_blocked": 0, "llm_no": 0, "cat1": 0, "cat2": 0}

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pid = int(row["id"])
            if pid in done_ids:
                continue

            paragraph = str(row["paragraph"])

            if not prefilter_broad(paragraph):
                stats["prefilter_blocked"] += 1
                record = {
                    "id": pid, "paragraph": paragraph,
                    "hybrid_is_hazard": "no", "category": "", "threat": ""
                }
                print(f"ID {pid}: prefilter blocked")
            else:
                system, user = build_hybrid_prompt(paragraph)
                result = call_llm(system, user)

                is_hazard = str(result.get("is_hazard", "no")).strip().lower()
                if is_hazard not in ["yes", "no"]:
                    is_hazard = "no"

                category = str(result.get("category", "")).strip()
                threat = str(result.get("threat", "")).strip()

                if is_hazard == "yes":
                    if category in ["1", "2"]:
                        stats[f"cat{category}"] += 1
                    else:
                        stats["cat1"] += 1  # default to cat 1
                    print(f"ID {pid}: YES (cat {category}) — {threat[:60]}")
                else:
                    stats["llm_no"] += 1
                    print(f"ID {pid}: no")

                record = {
                    "id": pid, "paragraph": paragraph,
                    "hybrid_is_hazard": is_hazard,
                    "category": category, "threat": threat
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(0.05)

    print(f"\nStats: {stats}")
    print(f"Saved to {OUT_PATH}")


def evaluate(df):
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in df.iterrows()]

    # Load all prediction sets
    def load_preds(path, field):
        preds = {}
        if path.exists():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        preds[int(obj["id"])] = obj
        return preds

    hybrid_preds = load_preds(OUT_PATH, "hybrid_is_hazard")
    guided_preds = load_preds(Path("data/processed/qwen_nothinking_predictions.jsonl"), "qwen_nothinking_is_hazard")
    llama_preds = load_preds(Path("data/processed/ollama_predictions.jsonl"), "ollama_is_hazard")
    openended_preds = load_preds(Path("data/processed/openended_predictions.jsonl"), "openended_is_hazard")

    y_hybrid = [1 if hybrid_preds.get(int(row["id"]), {}).get("hybrid_is_hazard", "no") == "yes" else 0
                for _, row in df.iterrows()]
    y_guided = [1 if guided_preds.get(int(row["id"]), {}).get("qwen_nothinking_is_hazard", "no") == "yes" else 0
                for _, row in df.iterrows()]
    y_llama = [1 if llama_preds.get(int(row["id"]), {}).get("ollama_is_hazard", "no").lower() == "yes" else 0
               for _, row in df.iterrows()]
    y_open = [1 if openended_preds.get(int(row["id"]), {}).get("openended_is_hazard", "no") == "yes" else 0
              for _, row in df.iterrows()]

    # Consensus: Llama + Hybrid
    y_consensus_hybrid = [1 if (l == 1 or h == 1) else 0 for l, h in zip(y_llama, y_hybrid)]

    def eval_print(name, yt, yp):
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{name:<50s} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {fp:>5d} {fn:>5d}")

    print(f"\n{'='*85}")
    print("COMPLETE COMPARISON: All Approaches")
    print(f"{'='*85}")
    print(f"{'Method':<50s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 85)
    eval_print("Llama 3.2 3B (guided only)", y_true, y_llama)
    eval_print("Qwen 397B (guided only)", y_true, y_guided)
    eval_print("Qwen 397B (open-ended, no rules)", y_true, y_open)
    eval_print("Qwen 397B (hybrid: rules + discovery)", y_true, y_hybrid)
    eval_print("Consensus: Llama + Guided Qwen", y_true, [1 if (l==1 or g==1) else 0 for l,g in zip(y_llama, y_guided)])
    eval_print("Consensus: Llama + Hybrid Qwen", y_true, y_consensus_hybrid)
    print("-" * 85)
    print(f"{'Bookworm (reported, approx.)':<50s} {'~0.560':>8s} {'~1.000':>8s} {'~0.720':>8s} {'—':>5s} {'—':>5s}")

    # Category breakdown
    print(f"\n{'='*85}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*85}")
    cat1_tp, cat1_fp, cat2_tp, cat2_fp = 0, 0, 0, 0
    cat2_novel = []

    for _, row in df.iterrows():
        pid = int(row["id"])
        hp = hybrid_preds.get(pid, {})
        gold = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
        pred = 1 if hp.get("hybrid_is_hazard", "no") == "yes" else 0
        cat = hp.get("category", "")

        if pred == 1 and cat == "1":
            if gold == 1:
                cat1_tp += 1
            else:
                cat1_fp += 1
        elif pred == 1 and cat == "2":
            if gold == 1:
                cat2_tp += 1
            else:
                cat2_fp += 1
                cat2_novel.append({
                    "id": pid,
                    "threat": hp.get("threat", ""),
                    "paragraph": str(row["paragraph"])[:80]
                })

    print(f"Category 1 (standard HIs): {cat1_tp} TP, {cat1_fp} FP")
    print(f"Category 2 (novel threats): {cat2_tp} TP, {cat2_fp} FP")

    if cat2_novel:
        print(f"\n--- Category 2: Novel threats beyond gold standard ---")
        for item in cat2_novel:
            print(f"  ID {item['id']}: {item['paragraph']}...")
            print(f"    Threat: {item['threat'][:70]}")
        print(f"\n  Total novel threats: {len(cat2_novel)}")
        print(f"  These need manual review — some may be genuine new vulnerabilities!")

    # Show what each approach uniquely catches
    print(f"\n{'='*85}")
    print("WHAT HYBRID CATCHES THAT GUIDED MISSES (on gold standard)")
    print(f"{'='*85}")
    for _, row in df.iterrows():
        pid = int(row["id"])
        gold = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
        g = guided_preds.get(pid, {}).get("qwen_nothinking_is_hazard", "no") == "yes"
        h = hybrid_preds.get(pid, {}).get("hybrid_is_hazard", "no") == "yes"
        if gold == 1 and h and not g:
            threat = hybrid_preds.get(pid, {}).get("threat", "")
            print(f"  ID {pid}: {str(row['paragraph'])[:80]}...")
            print(f"    Threat: {threat[:70]}")

    print(f"\nSaved to {OUT_PATH}")


def main():
    df = pd.read_csv(GOLD_PATH)

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(df)
    else:
        run(df)
        evaluate(df)


if __name__ == "__main__":
    main()