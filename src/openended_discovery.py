"""
PHASE B: Open-Ended Discovery Mode
====================================
Instead of telling the LLM what hazard indicators look like (guided mode),
we ask it to find ANY behavior that could be exploited by an attacker.

This answers the professor's key question: "Can LLMs discover things
we humans don't specify as rules?"

Comparison:
  Mode 1 (guided): "Here are 5 rules for what counts as a hazard. Follow them."
  Mode 2 (open-ended): "Find any exploitable behavior. No predefined rules."

We then compare:
  - What does Mode 2 find that Mode 1 misses?
  - What does Mode 1 find that Mode 2 misses?
  - Combined: if either mode says yes, what's the P/R/F1?
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time
import sys

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OUT_PATH = Path("data/processed/openended_predictions.jsonl")

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


def build_openended_prompt(paragraph):
    """
    Open-ended prompt: NO predefined hazard types, NO keyword lists, NO rules.
    Just ask the LLM to think like an attacker.
    """
    system = (
        "You are an adversarial security researcher analyzing telecom protocol specifications. "
        "Return ONLY JSON. No extra text."
    )
    user = f"""
Read this paragraph from an LTE protocol specification (3GPP TS 24.301).

Think like an attacker: Is there ANY behavior described here that could be
exploited to harm the network, disrupt service, compromise user privacy,
or bypass security mechanisms?

Consider:
- Could an attacker trigger this behavior deliberately?
- Could this behavior lead to denial of service?
- Could this behavior leak sensitive information?
- Could this behavior bypass authentication or security checks?
- Could this behavior cause inconsistent state between network entities?
- Are there any race conditions, timing issues, or edge cases?

Paragraph:
\"\"\"{paragraph}\"\"\"

If you identify ANY exploitable behavior, return:
{{"is_hazard":"yes","threat":"brief description of the threat"}}

If this paragraph describes only normal, non-exploitable behavior (definitions,
message formats, standard procedures with no security implications), return:
{{"is_hazard":"no","threat":""}}

Return ONLY the JSON object.
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


def run_openended(df):
    """Run open-ended discovery on all paragraphs (no prefilter!)"""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(OUT_PATH)

    if len(done_ids) >= len(df):
        print("Already complete. Skipping to evaluation.")
        return

    print(f"Running open-ended discovery mode...")
    print(f"NOTE: No prefilter applied — every paragraph goes to the LLM.")
    print(f"Already done: {len(done_ids)}, remaining: {len(df) - len(done_ids)}\n")

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pid = int(row["id"])
            if pid in done_ids:
                continue

            paragraph = str(row["paragraph"])

            system, user = build_openended_prompt(paragraph)
            result = call_llm(system, user)

            is_hazard = str(result.get("is_hazard", "no")).strip().lower()
            if is_hazard not in ["yes", "no"]:
                is_hazard = "no"

            threat = str(result.get("threat", "")).strip()

            record = {
                "id": pid,
                "paragraph": paragraph,
                "openended_is_hazard": is_hazard,
                "openended_threat": threat,
            }

            status = f"YES: {threat[:60]}" if is_hazard == "yes" else "no"
            print(f"ID {pid}: {status}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(0.05)

    print(f"\nDone. Saved to {OUT_PATH}")


def evaluate(df):
    """Compare guided mode vs open-ended mode"""
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in df.iterrows()]

    # Load open-ended predictions
    open_preds = {}
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    open_preds[int(obj["id"])] = obj

    if not open_preds:
        print("No open-ended predictions found. Run first.")
        return

    y_open = [1 if open_preds.get(int(row["id"]), {}).get("openended_is_hazard", "no") == "yes" else 0
              for _, row in df.iterrows()]

    # Load guided mode (A4 fewshot-3, our best prompt)
    guided_path = Path("data/processed/qwen_nothinking_predictions.jsonl")
    guided_preds = {}
    if guided_path.exists():
        with guided_path.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    guided_preds[int(obj["id"])] = obj

    y_guided = [1 if guided_preds.get(int(row["id"]), {}).get("qwen_nothinking_is_hazard", "no") == "yes" else 0
                for _, row in df.iterrows()]

    # Also load Llama for reference
    llama_path = Path("data/processed/ollama_predictions.jsonl")
    llama_preds = {}
    if llama_path.exists():
        with llama_path.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    llama_preds[int(obj["id"])] = obj

    y_llama = [1 if llama_preds.get(int(row["id"]), {}).get("ollama_is_hazard", "no").lower() == "yes" else 0
               for _, row in df.iterrows()]

    # Combined: guided OR open-ended
    y_combined = [1 if (g == 1 or o == 1) else 0 for g, o in zip(y_guided, y_open)]

    # Full consensus: Llama OR Qwen-guided OR open-ended
    y_full = [1 if (l == 1 or g == 1 or o == 1) else 0 for l, g, o in zip(y_llama, y_guided, y_open)]

    def eval_print(name, yt, yp):
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{name:<50s} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {fp:>5d} {fn:>5d}")

    print(f"\n{'='*85}")
    print("MODE COMPARISON: Guided vs Open-Ended Discovery")
    print(f"{'='*85}")
    print(f"{'Method':<50s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 85)
    eval_print("Llama 3.2 3B (guided, reference)", y_true, y_llama)
    eval_print("Qwen 397B (guided, rules+examples)", y_true, y_guided)
    eval_print("Qwen 397B (open-ended, no rules)", y_true, y_open)
    print("-" * 85)
    eval_print("Combined: Guided OR Open-ended", y_true, y_combined)
    eval_print("Full: Llama OR Guided OR Open-ended", y_true, y_full)

    # ── Detailed Analysis ──
    print(f"\n{'='*85}")
    print("DISCOVERY ANALYSIS")
    print(f"{'='*85}")

    # What open-ended found that guided missed
    print("\n--- Open-ended found, Guided missed (novel discoveries?) ---")
    novel_count = 0
    for _, row in df.iterrows():
        pid = int(row["id"])
        g = y_guided[_] if _ < len(y_guided) else 0
        o = y_open[_] if _ < len(y_open) else 0
        gold = y_true[_] if _ < len(y_true) else 0

        op = open_preds.get(pid, {})
        guided_yes = guided_preds.get(pid, {}).get("qwen_nothinking_is_hazard", "no") == "yes"
        open_yes = op.get("openended_is_hazard", "no") == "yes"

        if open_yes and not guided_yes:
            gold_label = "TRUE HI" if gold == 1 else "NOT in gold"
            threat = op.get("openended_threat", "")[:80]
            para = str(row["paragraph"])[:80]
            print(f"  ID {pid} [{gold_label}]: {para}...")
            print(f"    Threat: {threat}")
            novel_count += 1

    if novel_count == 0:
        print("  None — guided mode caught everything open-ended found")
    else:
        print(f"\n  Total: {novel_count} paragraphs found by open-ended but not guided")

    # What guided found that open-ended missed
    print("\n--- Guided found, Open-ended missed ---")
    missed_count = 0
    for _, row in df.iterrows():
        pid = int(row["id"])
        op = open_preds.get(pid, {})
        guided_yes = guided_preds.get(pid, {}).get("qwen_nothinking_is_hazard", "no") == "yes"
        open_yes = op.get("openended_is_hazard", "no") == "yes"
        gold = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0

        if guided_yes and not open_yes and gold == 1:
            para = str(row["paragraph"])[:100]
            print(f"  ID {pid}: {para}...")
            missed_count += 1

    if missed_count == 0:
        print("  None — open-ended caught everything guided found")
    else:
        print(f"\n  Total: {missed_count} true HIs caught by guided but missed by open-ended")

    # Novel threats (open-ended YES, not in gold standard)
    print("\n--- Potential NOVEL threats (open-ended YES, NOT in gold standard) ---")
    print("  These could be real hazards that your gold standard didn't capture!")
    novel_threats = 0
    for _, row in df.iterrows():
        pid = int(row["id"])
        op = open_preds.get(pid, {})
        open_yes = op.get("openended_is_hazard", "no") == "yes"
        gold = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0

        if open_yes and gold == 0:
            threat = op.get("openended_threat", "")[:80]
            para = str(row["paragraph"])[:80]
            print(f"  ID {pid}: {para}...")
            print(f"    Threat: {threat}")
            novel_threats += 1

    if novel_threats == 0:
        print("  None found")
    else:
        print(f"\n  Total: {novel_threats} potential novel threats — review these manually!")

    print(f"\nAll predictions saved to {OUT_PATH}")


def main():
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold file not found: {GOLD_PATH}")

    df = pd.read_csv(GOLD_PATH)

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(df)
    else:
        run_openended(df)
        evaluate(df)


if __name__ == "__main__":
    main()