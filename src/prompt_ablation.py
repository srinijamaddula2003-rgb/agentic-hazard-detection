"""
PROMPT ABLATION STUDY
======================
Tests 5 different prompting strategies on the same gold standard
to measure how prompt design affects HI detection performance.

Variants:
  A1: Pure zero-shot (no examples, no rules)
  A2: Guided zero-shot (rules only, no examples)
  A3: Few-shot 1 example
  A4: Few-shot 3 examples (current pipeline prompt)
  A5: Few-shot 5 examples

All use Qwen 3.5 397B with thinking OFF for fair comparison.
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time
import sys

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OUT_DIR = Path("data/processed/ablation")

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5"

# ── Prefilter (same for all variants for fair comparison) ────────────────────
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
PURE_BENIGN_MARKERS = [
    "is defined as", "refers to", "for the purposes of",
    "in this specification", "is used to describe", "abbreviation for",
]


def prefilter_maybe_hazard(paragraph: str) -> bool:
    t = paragraph.lower()
    if t.startswith("this message is sent by"):
        return False
    DESCRIPTION_STARTERS = [
        "the state ", "the substate ", "the purpose of the ",
        "this procedure is used", "is used to "
    ]
    if any(t.startswith(s) for s in DESCRIPTION_STARTERS):
        if not any(v in t for v in STRONG_RISKY):
            return False
    if any(b in t for b in PURE_BENIGN_MARKERS):
        if not any(k in t for k in STRONG_RISKY + MEDIUM_RISKY):
            return False
    has_cond = any(k in t for k in CONDITION_TRIGGERS)
    has_abnormal = any(k in t for k in ABNORMAL_TRIGGERS)
    has_strong = any(k in t for k in STRONG_RISKY)
    has_medium = any(k in t for k in MEDIUM_RISKY)
    if (has_cond or has_abnormal) and (has_strong or has_medium):
        return True
    if has_strong:
        return True
    return False


def call_llm(system: str, user: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 300,
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


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

def prompt_A1_pure_zero_shot(paragraph):
    """A1: Pure zero-shot — no examples, no rules"""
    system = (
        "You are a telecom protocol security analyst. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\"} or {\"is_hazard\":\"no\"}. "
        "No extra text."
    )
    user = f"""
Does this LTE specification paragraph contain a hazard indicator?

A hazard indicator is a statement that describes a security-relevant or
potentially dangerous operation in a telecom protocol specification.

Paragraph:
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def prompt_A2_guided_zero_shot(paragraph):
    """A2: Guided zero-shot — rules but no examples"""
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

=== RETURN "yes" ONLY if the paragraph describes: ===
1. Bearer context being DEACTIVATED or DELETED in response to a message/event
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials (GUTI, KSI, TAI list) being DELETED or INVALIDATED
4. Context being RESET or RELEASED due to a reject or failure
5. State transition to EMM-DEREGISTERED triggered by an error condition

=== RETURN "no" for: ===
1. Message format tables ("This message is sent by...", IEI tables)
2. State definitions ("EMM-NULL: The EPS capability is disabled")
3. Normal procedural steps ("The UE shall send an ATTACH REQUEST")
4. A REJECT message being SENT as a response (sending a reject is normal, not hazardous)
5. General descriptions of procedure purposes
6. Timer definitions, abbreviation lists, scope descriptions
7. Editor's notes

Paragraph to classify:
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def prompt_A3_fewshot_1(paragraph):
    """A3: Few-shot with 1 example"""
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

=== RETURN "yes" ONLY if the paragraph describes: ===
1. Bearer context being DEACTIVATED or DELETED in response to a message/event
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials (GUTI, KSI, TAI list) being DELETED or INVALIDATED
4. Context being RESET or RELEASED due to a reject or failure
5. State transition to EMM-DEREGISTERED triggered by an error condition

=== RETURN "no" for: ===
1. Message format tables, state definitions, normal procedural steps
2. A REJECT message being SENT as a response
3. General descriptions of procedure purposes
4. Timer definitions, abbreviation lists, scope descriptions

=== Example: ===

HAZARD (yes):
"Upon receipt of an AUTHENTICATION REJECT message, the mobile station shall set the
update status to EU3, delete the stored GUTI, TAI list, and KSI."
Reason: Credential deletion triggered by a specific message

Paragraph to classify:
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def prompt_A4_fewshot_3(paragraph):
    """A4: Few-shot with 3 examples (current pipeline prompt)"""
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

=== RETURN "yes" ONLY if the paragraph describes: ===
1. Bearer context being DEACTIVATED or DELETED in response to a message/event
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials (GUTI, KSI, TAI list) being DELETED or INVALIDATED
4. Context being RESET or RELEASED due to a reject or failure
5. State transition to EMM-DEREGISTERED triggered by an error condition

=== RETURN "no" for: ===
1. Message format tables ("This message is sent by...", IEI tables)
2. State definitions ("EMM-NULL: The EPS capability is disabled")
3. Normal procedural steps ("The UE shall send an ATTACH REQUEST")
4. A REJECT message being SENT as a response (sending a reject is normal, not hazardous)
5. General descriptions of procedure purposes
6. Timer definitions, abbreviation lists, scope descriptions
7. Editor's notes

=== Examples: ===

HAZARD (yes):
"Upon receipt of an AUTHENTICATION REJECT message, the mobile station shall set the
update status to EU3, delete the stored GUTI, TAI list, and KSI."
Reason: Credential deletion triggered by a specific message

NOT A HAZARD (no):
"If the bearer resource allocation requested cannot be accepted by the network, the
MME shall send a BEARER RESOURCE ALLOCATION REJECT message to the UE."
Reason: Just sending a reject message, no destructive action on context/credentials

NOT A HAZARD (no):
"The detach procedure is used by the UE to inform the network that it does not want
to access the EPS any longer."
Reason: General description of purpose, no specific destructive operation

Paragraph to classify:
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def prompt_A5_fewshot_5(paragraph):
    """A5: Few-shot with 5 examples"""
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

=== RETURN "yes" ONLY if the paragraph describes: ===
1. Bearer context being DEACTIVATED or DELETED in response to a message/event
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials (GUTI, KSI, TAI list) being DELETED or INVALIDATED
4. Context being RESET or RELEASED due to a reject or failure
5. State transition to EMM-DEREGISTERED triggered by an error condition

=== RETURN "no" for: ===
1. Message format tables ("This message is sent by...", IEI tables)
2. State definitions ("EMM-NULL: The EPS capability is disabled")
3. Normal procedural steps ("The UE shall send an ATTACH REQUEST")
4. A REJECT message being SENT as a response (sending a reject is normal, not hazardous)
5. General descriptions of procedure purposes
6. Timer definitions, abbreviation lists, scope descriptions
7. Editor's notes

=== Examples: ===

HAZARD (yes):
"Upon receipt of an AUTHENTICATION REJECT message, the mobile station shall set the
update status to EU3, delete the stored GUTI, TAI list, and KSI."
Reason: Credential deletion triggered by a specific message

HAZARD (yes):
"When receiving the DETACH REQUEST message and the Detach type IE indicates re-attach
required, the UE shall deactivate the EPS bearer context(s) including the EPS default
bearer context locally."
Reason: Bearer deactivation triggered by a message

HAZARD (yes):
"Upon receipt of the ACTIVATE DEDICATED EPS BEARER CONTEXT REJECT message in state
BEARER CONTEXT ACTIVE PENDING, the MME shall enter the state BEARER CONTEXT INACTIVE
and abort the dedicated EPS bearer context activation procedure."
Reason: State transition + procedure abort triggered by a reject message

NOT A HAZARD (no):
"If the bearer resource allocation requested cannot be accepted by the network, the
MME shall send a BEARER RESOURCE ALLOCATION REJECT message to the UE."
Reason: Just sending a reject message, no destructive action on context/credentials

NOT A HAZARD (no):
"The detach procedure is used by the UE to inform the network that it does not want
to access the EPS any longer."
Reason: General description of purpose, no specific destructive operation

Paragraph to classify:
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

VARIANTS = {
    "A1_zero_shot": prompt_A1_pure_zero_shot,
    "A2_guided_zero_shot": prompt_A2_guided_zero_shot,
    "A3_fewshot_1": prompt_A3_fewshot_1,
    "A4_fewshot_3": prompt_A4_fewshot_3,
    "A5_fewshot_5": prompt_A5_fewshot_5,
}


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


def run_variant(variant_name, prompt_fn, df):
    out_path = OUT_DIR / f"{variant_name}_predictions.jsonl"
    done_ids = load_done_ids(out_path)
    skipped = len(done_ids)

    if skipped >= len(df):
        print(f"  Already complete ({skipped} predictions). Skipping.")
        return

    print(f"  Resuming from {skipped} done predictions...")

    with out_path.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pid = int(row["id"])
            if pid in done_ids:
                continue

            paragraph = str(row["paragraph"])

            # Same prefilter for all variants
            if not prefilter_maybe_hazard(paragraph):
                is_hazard = "no"
            else:
                system, user = prompt_fn(paragraph)
                cls = call_llm(system, user)
                is_hazard = str(cls.get("is_hazard", "no")).strip().lower()
                if is_hazard not in ["yes", "no"]:
                    is_hazard = "no"

            record = {
                "id": pid,
                "paragraph": paragraph,
                "is_hazard": is_hazard,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(0.05)

    print(f"  Done. Saved to {out_path}")


def evaluate_all(df):
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in df.iterrows()]

    print(f"\n{'='*75}")
    print("PROMPT ABLATION STUDY RESULTS")
    print(f"{'='*75}")
    print(f"{'Variant':<35s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 75)

    for variant_name in VARIANTS:
        out_path = OUT_DIR / f"{variant_name}_predictions.jsonl"
        if not out_path.exists():
            print(f"{variant_name:<35s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>5s} {'—':>5s}")
            continue

        preds = {}
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    preds[int(obj["id"])] = obj

        y_pred = [1 if preds.get(int(row["id"]), {}).get("is_hazard", "no") == "yes" else 0
                  for _, row in df.iterrows()]

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        desc = {
            "A1_zero_shot": "A1: Pure zero-shot (no rules/examples)",
            "A2_guided_zero_shot": "A2: Guided zero-shot (rules only)",
            "A3_fewshot_1": "A3: Few-shot (1 example)",
            "A4_fewshot_3": "A4: Few-shot (3 examples) [current]",
            "A5_fewshot_5": "A5: Few-shot (5 examples)",
        }
        name = desc.get(variant_name, variant_name)
        print(f"{name:<35s} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {fp:>5d} {fn:>5d}")

    # Also show Llama baseline for reference
    llama_path = Path("data/processed/ollama_predictions.jsonl")
    if llama_path.exists():
        preds = {}
        with llama_path.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    preds[int(obj["id"])] = obj
        y_pred = [1 if preds.get(int(row["id"]), {}).get("ollama_is_hazard", "no").lower() == "yes" else 0
                  for _, row in df.iterrows()]
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print("-" * 75)
        print(f"{'Llama 3.2 3B (reference)':<35s} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {fp:>5d} {fn:>5d}")


def main():
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold file not found: {GOLD_PATH}")

    df = pd.read_csv(GOLD_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check which variant to run
    if len(sys.argv) > 1:
        variant = sys.argv[1]
        if variant == "eval":
            evaluate_all(df)
            return
        if variant not in VARIANTS:
            print(f"Unknown variant: {variant}")
            print(f"Available: {', '.join(VARIANTS.keys())}, eval")
            return
        print(f"Running variant: {variant}")
        run_variant(variant, VARIANTS[variant], df)
        evaluate_all(df)
    else:
        # Run all variants sequentially
        for variant_name, prompt_fn in VARIANTS.items():
            print(f"\n{'='*60}")
            print(f"Running: {variant_name}")
            print(f"{'='*60}")
            run_variant(variant_name, prompt_fn, df)

        evaluate_all(df)


if __name__ == "__main__":
    main()