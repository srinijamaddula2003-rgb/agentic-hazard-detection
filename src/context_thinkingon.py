"""
PHASE C: Context Engineering + Thinking ON
=============================================
Question: Can context engineering reduce the noise from thinking mode?

Previous findings:
- Thinking OFF + no context: F1=0.786 (baseline)
- Thinking OFF + context eng: F1=0.829 (perfect recall, 7 FP)
- Thinking ON + no context: F1=0.733 (worse — thinking adds noise)
- Thinking ON + context eng: ??? (THIS EXPERIMENT)

Hypothesis: Thinking mode overthinks when it lacks context. Providing
surrounding paragraphs and section metadata should ground the reasoning
and prevent speculation, potentially making thinking mode productive.

Run this with thinking ON server:
  Server WITHOUT --chat-template-kwargs (thinking defaults to ON)
  OR with --chat-template-kwargs '{"enable_thinking":true}'
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time
import sys

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OUT_PATH = Path("data/processed/context_thinking_predictions.jsonl")

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

HIGH_RISK_SECTIONS = [
    "abnormal", "not accepted", "reject", "failure", "error",
    "authentication not accepted", "security mode",
    "integrity checking", "detach", "deregistration"
]


def call_llm(system: str, user: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 2000,  # Need more tokens for thinking + answer
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(LLAMA_CPP_URL, json=payload, timeout=900)
    r.raise_for_status()

    msg = r.json()["choices"][0]["message"]
    text = msg.get("content", "").strip()
    reasoning = msg.get("reasoning_content", "").strip()

    # If content is empty (thinking used all tokens), check reasoning
    if not text and reasoning:
        json_match = re.search(r'\{[^{}]*"is_hazard"[^{}]*\}', reasoning)
        if json_match:
            text = json_match.group(0)

    # Strip thinking tags if present
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


def compute_risk_score(paragraph: str, section_title: str) -> int:
    t = paragraph.lower()
    score = 0

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

    st = section_title.lower()
    if any(h in st for h in HIGH_RISK_SECTIONS):
        score += 3

    return score


def get_surrounding_context(gold_rows, target_idx, window=2):
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
3. Normal successful procedure steps with no destructive side effects
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
    gold_rows = df.to_dict('records')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(OUT_PATH)

    # Determine which paragraphs need context engineering
    # Same approach as step3: all paragraphs with risk score >= 3
    # Plus: send ALL paragraphs through (no prefilter) for fair comparison with thinking OFF
    
    print("=" * 60)
    print("PHASE C: Context Engineering + Thinking ON")
    print("=" * 60)
    print("IMPORTANT: Ensure server is running with thinking ENABLED")
    print(f"Already done: {len(done_ids)}, remaining: {len(df) - len(done_ids)}\n")

    stats = {"low_risk_skip": 0, "llm_yes": 0, "llm_no": 0}

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for idx, row in enumerate(gold_rows):
            pid = int(row["id"])
            if pid in done_ids:
                continue

            paragraph = str(row["paragraph"])
            section_id = str(row.get("section_id", ""))
            section_title = str(row.get("section_title", ""))

            # Compute risk score
            score = compute_risk_score(paragraph, section_title)

            if score < 3:
                # Low risk — skip LLM call, classify as no
                stats["low_risk_skip"] += 1
                record = {
                    "id": pid, "paragraph": paragraph,
                    "context_thinking_is_hazard": "no",
                    "risk_score": score,
                }
                print(f"ID {pid}: skip (score={score})")
            else:
                # Get surrounding context
                before_paras, after_paras = get_surrounding_context(gold_rows, idx, window=2)

                # Classify with context + thinking ON
                sys_c, usr_c = build_context_prompt(
                    paragraph, section_id, section_title,
                    before_paras, after_paras
                )
                result = call_llm(sys_c, usr_c)
                is_hazard = str(result.get("is_hazard", "no")).lower()
                if is_hazard not in ["yes", "no"]:
                    is_hazard = "no"

                if is_hazard == "yes":
                    stats["llm_yes"] += 1
                    print(f"ID {pid}: YES (score={score}, section: {section_title[:40]})")
                else:
                    stats["llm_no"] += 1
                    print(f"ID {pid}: no (score={score})")

                record = {
                    "id": pid, "paragraph": paragraph,
                    "context_thinking_is_hazard": is_hazard,
                    "risk_score": score,
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(0.05)

    print(f"\nStats: {stats}")
    print(f"Saved to {OUT_PATH}")


def evaluate(df):
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in df.iterrows()]

    def load_preds(path, field):
        preds = {}
        if path.exists():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        preds[int(obj["id"])] = obj
        return [1 if preds.get(int(row["id"]), {}).get(field, "no").lower() == "yes" else 0
                for _, row in df.iterrows()]

    y_ctx_thinking = load_preds(OUT_PATH, "context_thinking_is_hazard")
    y_ctx_nothinking = load_preds(Path("data/processed/context_eng_predictions.jsonl"), "final_is_hazard")
    y_vanilla_thinking = load_preds(Path("data/processed/qwen_thinking_predictions.jsonl"), "qwen_thinking_is_hazard")
    y_vanilla_nothinking = load_preds(Path("data/processed/qwen_nothinking_predictions.jsonl"), "qwen_nothinking_is_hazard")
    y_llama = load_preds(Path("data/processed/ollama_predictions.jsonl"), "ollama_is_hazard")

    # Consensus combinations
    y_consensus_base = [1 if (l == 1 or n == 1) else 0
                        for l, n in zip(y_llama, y_vanilla_nothinking)]
    y_consensus_ctx_think = [1 if (l == 1 or n == 1 or ct == 1) else 0
                             for l, n, ct in zip(y_llama, y_vanilla_nothinking, y_ctx_thinking)]

    def eval_print(name, yt, yp):
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{name:<55s} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {fp:>5d} {fn:>5d}")

    print(f"\n{'='*90}")
    print("PHASE C: THINKING MODE COMPARISON WITH CONTEXT ENGINEERING")
    print(f"{'='*90}")
    print(f"{'Method':<55s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 90)
    print("--- Vanilla (no context) ---")
    eval_print("Qwen 397B thinking OFF (vanilla)", y_true, y_vanilla_nothinking)
    eval_print("Qwen 397B thinking ON (vanilla)", y_true, y_vanilla_thinking)
    print("--- With Context Engineering ---")
    eval_print("Qwen 397B thinking OFF + context", y_true, y_ctx_nothinking)
    eval_print("Qwen 397B thinking ON + context (THIS RUN)", y_true, y_ctx_thinking)
    print("--- Consensus Combinations ---")
    eval_print("Consensus base (Llama + Qwen nothinking)", y_true, y_consensus_base)
    eval_print("Consensus + context thinking ON", y_true, y_consensus_ctx_think)
    print("-" * 90)
    print(f"{'Bookworm (reported, approx.)':<55s} {'~0.560':>8s} {'~1.000':>8s} {'~0.720':>8s} {'—':>5s} {'—':>5s}")

    # Show what context+thinking catches vs misses
    print(f"\n--- Context+Thinking catches that vanilla misses ---")
    for i, row in df.iterrows():
        pid = int(row["id"])
        gold = y_true[i]
        vanilla = y_vanilla_nothinking[i]
        ctx_think = y_ctx_thinking[i]
        if gold == 1 and ctx_think == 1 and vanilla == 0:
            print(f"  ID {pid}: {str(row['paragraph'])[:100]}...")

    print(f"\n--- Context+Thinking FALSE POSITIVES ---")
    fp_count = 0
    for i, row in df.iterrows():
        pid = int(row["id"])
        gold = y_true[i]
        ctx_think = y_ctx_thinking[i]
        if gold == 0 and ctx_think == 1:
            print(f"  ID {pid}: {str(row['paragraph'])[:80]}...")
            fp_count += 1
    if fp_count == 0:
        print("  None!")

    print(f"\n--- Still missed by Context+Thinking ---")
    fn_count = 0
    for i, row in df.iterrows():
        pid = int(row["id"])
        gold = y_true[i]
        ctx_think = y_ctx_thinking[i]
        if gold == 1 and ctx_think == 0:
            print(f"  ID {pid}: {str(row['paragraph'])[:100]}...")
            fn_count += 1
    if fn_count == 0:
        print("  None! Perfect recall!")


def main():
    df = pd.read_csv(GOLD_PATH)

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(df)
    else:
        run(df)
        evaluate(df)


if __name__ == "__main__":
    main()