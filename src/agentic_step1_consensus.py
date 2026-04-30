"""
AGENTIC STEP 1: Multi-Model Consensus
======================================
Combines Llama 3.2 3B and Qwen 3.5 397B predictions using different
voting strategies to improve recall without sacrificing precision.

No new LLM calls needed — just combines existing prediction files.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OLLAMA_PATH = Path("data/processed/ollama_predictions.jsonl")
QWEN_PATH = Path("data/processed/qwen_predictions.jsonl")
OUT_PATH = Path("data/processed/consensus_predictions.jsonl")


def load_preds(path, prefix):
    preds = {}
    if not path.exists():
        print(f"WARNING: {path} not found")
        return preds
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = int(obj["id"])
            is_hazard = str(obj.get(f"{prefix}_is_hazard", "no")).strip().lower()
            preds[pid] = {
                "is_hazard": 1 if is_hazard == "yes" else 0,
                "condition": obj.get(f"{prefix}_condition", ""),
                "operation": obj.get(f"{prefix}_operation", ""),
                "state": obj.get(f"{prefix}_state", ""),
                "hazard_type": obj.get(f"{prefix}_hazard_type", ""),
            }
    return preds


def evaluate(name, y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    fp = sum(1 for t, pr in zip(y_true, y_pred) if t == 0 and pr == 1)
    fn = sum(1 for t, pr in zip(y_true, y_pred) if t == 1 and pr == 0)
    return p, r, f, fp, fn


def main():
    gold = pd.read_csv(GOLD_PATH)
    llama_preds = load_preds(OLLAMA_PATH, "ollama")
    qwen_preds = load_preds(QWEN_PATH, "qwen")

    y_true = []
    y_llama = []
    y_qwen = []
    y_union = []        # Either model says yes → yes (maximize recall)
    y_intersection = [] # Both models say yes → yes (maximize precision)

    for _, row in gold.iterrows():
        pid = int(row["id"])
        gold_label = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
        y_true.append(gold_label)

        llama = llama_preds.get(pid, {}).get("is_hazard", 0)
        qwen = qwen_preds.get(pid, {}).get("is_hazard", 0)

        y_llama.append(llama)
        y_qwen.append(qwen)
        y_union.append(1 if (llama == 1 or qwen == 1) else 0)
        y_intersection.append(1 if (llama == 1 and qwen == 1) else 0)

    # Print comparison table
    print("=" * 75)
    print("AGENTIC STEP 1: Multi-Model Consensus Results")
    print("=" * 75)
    print(f"{'Method':<35s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'FP':>5s} {'FN':>5s}")
    print("-" * 75)

    for name, y_pred in [
        ("Llama 3.2 3B (baseline)", y_llama),
        ("Qwen 3.5 397B (baseline)", y_qwen),
        ("Consensus: UNION (either=yes)", y_union),
        ("Consensus: INTERSECTION (both=yes)", y_intersection),
    ]:
        p, r, f, fp, fn = evaluate(name, y_true, y_pred)
        print(f"{name:<35s} {p:>10.3f} {r:>10.3f} {f:>10.3f} {fp:>5d} {fn:>5d}")

    # Detailed analysis of union strategy
    print("\n" + "=" * 75)
    print("UNION STRATEGY ANALYSIS")
    print("=" * 75)

    print("\n--- Gains: Caught by union but missed by individual models ---")
    gains = 0
    for i, row in gold.iterrows():
        pid = int(row["id"])
        gold_label = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
        llama = llama_preds.get(pid, {}).get("is_hazard", 0)
        qwen = qwen_preds.get(pid, {}).get("is_hazard", 0)
        union = 1 if (llama == 1 or qwen == 1) else 0

        if gold_label == 1 and union == 1:
            if llama == 0 or qwen == 0:
                saved_by = "Qwen only" if llama == 0 else "Llama only"
                para = str(row["paragraph"])[:100]
                print(f"  ID {pid} ({saved_by}): {para}...")
                gains += 1

    print(f"\nTotal rescued paragraphs: {gains}")

    print("\n--- Still missed by both models ---")
    still_missed = 0
    for i, row in gold.iterrows():
        pid = int(row["id"])
        gold_label = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
        llama = llama_preds.get(pid, {}).get("is_hazard", 0)
        qwen = qwen_preds.get(pid, {}).get("is_hazard", 0)

        if gold_label == 1 and llama == 0 and qwen == 0:
            para = str(row["paragraph"])[:100]
            print(f"  ID {pid}: {para}...")
            still_missed += 1

    print(f"\nStill missed: {still_missed} — these need Reflexion or Context Engineering")

    # Save union predictions
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in gold.iterrows():
            pid = int(row["id"])
            llama = llama_preds.get(pid, {})
            qwen = qwen_preds.get(pid, {})
            llama_yes = llama.get("is_hazard", 0) == 1
            qwen_yes = qwen.get("is_hazard", 0) == 1
            is_hazard = "yes" if (llama_yes or qwen_yes) else "no"

            # For extraction details, prefer whichever model said yes
            # If both said yes, prefer Qwen (stronger reasoning)
            if qwen_yes:
                source = qwen
                source_model = "qwen"
            elif llama_yes:
                source = llama
                source_model = "llama"
            else:
                source = {"condition": "", "operation": "", "state": "", "hazard_type": ""}
                source_model = "none"

            record = {
                "id": pid,
                "paragraph": str(row["paragraph"]),
                "consensus_is_hazard": is_hazard,
                "consensus_condition": source.get("condition", ""),
                "consensus_operation": source.get("operation", ""),
                "consensus_state": source.get("state", ""),
                "consensus_hazard_type": source.get("hazard_type", ""),
                "source_model": source_model,
                "llama_said": "yes" if llama_yes else "no",
                "qwen_said": "yes" if qwen_yes else "no",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved consensus predictions to {OUT_PATH}")


if __name__ == "__main__":
    main()