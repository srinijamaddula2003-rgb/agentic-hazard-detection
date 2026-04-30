"""
Compare all vanilla LLM baselines with controlled thinking settings.
Run AFTER both qwen_nothinking and qwen_thinking prediction files exist.
"""
from pathlib import Path
import json
import pandas as pd

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OLLAMA_PATH = Path("data/processed/ollama_predictions.jsonl")
QWEN_NOTHINKING_PATH = Path("data/processed/qwen_nothinking_predictions.jsonl")
QWEN_THINKING_PATH = Path("data/processed/qwen_thinking_predictions.jsonl")
QWEN_ORIGINAL_PATH = Path("data/processed/qwen_predictions.jsonl")


def load_preds(path, prefix):
    preds = {}
    if not path.exists():
        return preds
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = int(obj["id"])
            # Try the specific prefix first, then fall back
            val = obj.get(f"{prefix}_is_hazard", obj.get("qwen_is_hazard", "no"))
            preds[pid] = 1 if str(val).strip().lower() == "yes" else 0
    return preds


def eval_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1, fp, fn


def main():
    gold = pd.read_csv(GOLD_PATH)
    y_true = [1 if str(row["is_hazard"]).strip().lower() == "yes" else 0
              for _, row in gold.iterrows()]

    # Load all predictions
    llama = load_preds(OLLAMA_PATH, "ollama")
    qwen_orig = load_preds(QWEN_ORIGINAL_PATH, "qwen")
    qwen_nothink = load_preds(QWEN_NOTHINKING_PATH, "qwen_nothinking")
    qwen_think = load_preds(QWEN_THINKING_PATH, "qwen_thinking")

    def get_labels(preds):
        return [preds.get(int(row["id"]), 0) for _, row in gold.iterrows()]

    print("=" * 85)
    print("VANILLA LLM BASELINES — CONTROLLED THINKING COMPARISON")
    print("=" * 85)
    print(f"{'Method':<45s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 85)

    methods = [("Llama 3.2 3B (local, Ollama)", llama)]

    if qwen_orig:
        methods.append(("Qwen 397B (original run, thinking=?)", qwen_orig))
    if qwen_nothink:
        methods.append(("Qwen 397B (thinking OFF)", qwen_nothink))
    if qwen_think:
        methods.append(("Qwen 397B (thinking ON)", qwen_think))

    for name, preds in methods:
        if not preds:
            print(f"{name:<45s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>5s} {'—':>5s}")
            continue
        y_pred = get_labels(preds)
        p, r, f, fp, fn = eval_metrics(y_true, y_pred)
        print(f"{name:<45s} {p:>8.3f} {r:>8.3f} {f:>8.3f} {fp:>5d} {fn:>5d}")

    # Consensus variations
    print("\n" + "=" * 85)
    print("CONSENSUS COMBINATIONS")
    print("=" * 85)
    print(f"{'Method':<45s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print("-" * 85)

    y_llama = get_labels(llama)

    for qwen_name, qwen_preds in [("Qwen no-thinking", qwen_nothink),
                                    ("Qwen thinking", qwen_think)]:
        if not qwen_preds:
            continue
        y_qwen = get_labels(qwen_preds)
        y_union = [1 if (l == 1 or q == 1) else 0 for l, q in zip(y_llama, y_qwen)]
        p, r, f, fp, fn = eval_metrics(y_true, y_union)
        print(f"Union: Llama + {qwen_name:<28s} {p:>8.3f} {r:>8.3f} {f:>8.3f} {fp:>5d} {fn:>5d}")

    # Show disagreements between thinking modes
    if qwen_nothink and qwen_think:
        print("\n" + "=" * 85)
        print("THINKING ON vs OFF — DISAGREEMENTS")
        print("=" * 85)
        disagree = 0
        for _, row in gold.iterrows():
            pid = int(row["id"])
            nt = qwen_nothink.get(pid, 0)
            th = qwen_think.get(pid, 0)
            if nt != th:
                gold_label = "YES" if str(row["is_hazard"]).strip().lower() == "yes" else "NO"
                nt_label = "YES" if nt == 1 else "NO"
                th_label = "YES" if th == 1 else "NO"
                para = str(row["paragraph"])[:100]
                print(f"  ID {pid}: Gold={gold_label} | NoThink={nt_label} | Think={th_label}")
                print(f"    {para}...")
                disagree += 1
        print(f"\nTotal disagreements: {disagree}")
        if disagree == 0:
            print("  Both modes gave identical results!")


if __name__ == "__main__":
    main()