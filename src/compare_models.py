"""
Compare Llama 3.2 3B (Ollama) vs Qwen 3.5 397B (llama.cpp) predictions
Run this AFTER both prediction files exist.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from baseline_keyword import keyword_classify

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OLLAMA_PATH = Path("data/processed/ollama_predictions.jsonl")
QWEN_PATH = Path("data/processed/qwen_predictions.jsonl")

def load_preds(path, field_prefix):
    preds = {}
    if not path.exists():
        return preds
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            preds[int(obj["id"])] = obj
    return preds

def get_pred_label(preds, pid, field_prefix):
    obj = preds.get(pid)
    if obj is None:
        return None
    val = str(obj.get(f"{field_prefix}_is_hazard", "no")).strip().lower()
    return 1 if val == "yes" else 0

def main():
    gold = pd.read_csv(GOLD_PATH)
    ollama_preds = load_preds(OLLAMA_PATH, "ollama")
    qwen_preds = load_preds(QWEN_PATH, "qwen")

    y_true = []
    y_keyword = []
    y_ollama = []
    y_qwen = []

    for _, row in gold.iterrows():
        pid = int(row["id"])
        gold_label = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
        y_true.append(gold_label)
        y_keyword.append(1 if keyword_classify(str(row["paragraph"])) == "yes" else 0)

        ol = get_pred_label(ollama_preds, pid, "ollama")
        y_ollama.append(ol if ol is not None else 0)

        qw = get_pred_label(qwen_preds, pid, "qwen")
        y_qwen.append(qw if qw is not None else 0)

    print("=" * 70)
    print("COMPARISON: Keyword vs Llama 3.2 3B vs Qwen 3.5 397B")
    print("=" * 70)
    print(f"{'Method':<30s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'FP':>6s} {'FN':>6s}")
    print("-" * 70)

    for name, y_pred in [("Keyword baseline", y_keyword),
                          ("Llama 3.2 3B (Ollama)", y_ollama),
                          ("Qwen 3.5 397B (llama.cpp)", y_qwen)]:
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)
        fp = sum(1 for t, pr in zip(y_true, y_pred) if t == 0 and pr == 1)
        fn = sum(1 for t, pr in zip(y_true, y_pred) if t == 1 and pr == 0)
        print(f"{name:<30s} {p:>10.3f} {r:>10.3f} {f:>10.3f} {fp:>6d} {fn:>6d}")

    # Show where Qwen and Llama disagree
    print("\n--- Cases where Qwen and Llama DISAGREE ---")
    disagree_count = 0
    for i, row in gold.iterrows():
        pid = int(row["id"])
        ol = get_pred_label(ollama_preds, pid, "ollama")
        qw = get_pred_label(qwen_preds, pid, "qwen")
        if ol is not None and qw is not None and ol != qw:
            gold_label = "YES" if str(row["is_hazard"]).strip().lower() == "yes" else "NO"
            ol_label = "YES" if ol == 1 else "NO"
            qw_label = "YES" if qw == 1 else "NO"
            para = str(row["paragraph"])[:100]
            print(f"  ID {pid}: Gold={gold_label} | Llama={ol_label} | Qwen={qw_label}")
            print(f"    {para}...")
            disagree_count += 1
    print(f"\nTotal disagreements: {disagree_count}")

if __name__ == "__main__":
    main()