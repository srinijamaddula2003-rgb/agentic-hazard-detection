from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

GOLD_PATH = Path("data/processed/annotation_sample.csv")
PRED_PATH = Path("data/processed/qwen_predictions.jsonl")

def load_preds(path: Path):
    preds = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            preds[int(obj["id"])] = obj
    return preds

def main():
    gold = pd.read_csv(GOLD_PATH)
    preds = load_preds(PRED_PATH)

    y_true, y_pred = [], []
    missing = 0

    for _, row in gold.iterrows():
        pid = int(row["id"])
        gold_label = 1 if str(row["is_hazard"]).strip().lower() == "yes" else 0

        pred_obj = preds.get(pid)
        if pred_obj is None:
            missing += 1
            continue

        pred_label = 1 if str(pred_obj["qwen_is_hazard"]).strip().lower() == "yes" else 0

        y_true.append(gold_label)
        y_pred.append(pred_label)

    print("=" * 60)
    print("QWEN 3.5 (397B) EVALUATION RESULTS")
    print("=" * 60)
    print(f"Gold rows: {len(gold)}")
    print(f"Pred rows found: {len(y_true)} (missing {missing})")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred):.3f}")
    print("Confusion matrix [ [TN FP], [FN TP] ]:")
    print(confusion_matrix(y_true, y_pred))

    print("\n--- FP paragraphs (model said YES, gold says NO) ---")
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        pid = int(gold.iloc[i]["id"])
        if yt == 0 and yp == 1:
            para = preds[pid]["paragraph"][:120]
            print(f"  ID {pid}: {para}...")

    print("\n--- FN paragraphs (model said NO, gold says YES) ---")
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        pid = int(gold.iloc[i]["id"])
        if yt == 1 and yp == 0:
            para = gold.iloc[i]["paragraph"][:120]
            print(f"  ID {pid}: {para}...")

if __name__ == "__main__":
    main()