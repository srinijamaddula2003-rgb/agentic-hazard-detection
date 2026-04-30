from pathlib import Path
import json
import csv
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    print("=" * 60)
    print("LTE HAZARD INDICATOR DISCOVERY — EVALUATION REPORT")
    print("=" * 60)

    # 1. Data stats
    paras = [json.loads(l) for l in open("data/processed/TS_24.301_paragraphs.jsonl") if l.strip()]
    tables = sum(1 for p in paras if p.get('is_table'))
    print(f"\n1. DATA")
    print(f"   Spec: TS 24.301 V0.1.0 (Release 8)")
    print(f"   Total paragraphs: {len(paras)}")
    print(f"   Content paragraphs: {len(paras) - tables}")
    print(f"   Table paragraphs: {tables}")

    # 2. Annotation stats
    with open("data/processed/annotation_sample.csv", encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))
    yes_rows = [r for r in gold_rows if r['is_hazard'].strip().lower() == 'yes']
    types = Counter(r['hazard_type'] for r in yes_rows)
    print(f"\n2. GOLD ANNOTATIONS")
    print(f"   Annotated paragraphs: {len(gold_rows)}")
    print(f"   Hazard indicators: {len(yes_rows)} ({len(yes_rows)/len(gold_rows)*100:.1f}%)")
    print(f"   By type: {dict(types)}")

    # 3. Classification comparison
    print(f"\n3. CLASSIFICATION RESULTS")
    print(f"   {'Method':<30s} {'P':>8s} {'R':>8s} {'F1':>8s} {'FP':>5s} {'FN':>5s}")
    print(f"   {'-'*65}")

    y_true = [1 if r['is_hazard'].strip().lower() == 'yes' else 0 for r in gold_rows]

    # Keyword baseline
    from baseline_keyword import keyword_classify
    y_kw = [1 if keyword_classify(r['paragraph']) == 'yes' else 0 for r in gold_rows]
    p_kw = precision_score(y_true, y_kw)
    r_kw = recall_score(y_true, y_kw)
    f_kw = f1_score(y_true, y_kw)
    fp_kw = sum(1 for t, p in zip(y_true, y_kw) if t == 0 and p == 1)
    fn_kw = sum(1 for t, p in zip(y_true, y_kw) if t == 1 and p == 0)
    print(f"   {'Keyword-only':<30s} {p_kw:>8.3f} {r_kw:>8.3f} {f_kw:>8.3f} {fp_kw:>5d} {fn_kw:>5d}")

    # Ollama baseline
    pred_path = Path("data/processed/ollama_predictions.jsonl")
    if pred_path.exists():
        preds = {}
        with pred_path.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    preds[int(obj['id'])] = obj

        y_ollama = []
        for r in gold_rows:
            pid = int(r['id'])
            p = preds.get(pid)
            y_ollama.append(1 if p and p['ollama_is_hazard'].lower() == 'yes' else 0)

        p_ol = precision_score(y_true, y_ollama)
        r_ol = recall_score(y_true, y_ollama)
        f_ol = f1_score(y_true, y_ollama)
        fp_ol = sum(1 for t, p in zip(y_true, y_ollama) if t == 0 and p == 1)
        fn_ol = sum(1 for t, p in zip(y_true, y_ollama) if t == 1 and p == 0)
        print(f"   {'Llama 3.2:3b (pipeline)':<30s} {p_ol:>8.3f} {r_ol:>8.3f} {f_ol:>8.3f} {fp_ol:>5d} {fn_ol:>5d}")

    # 4. State recovery
    hi_path = Path("data/processed/hazard_indicators_structured.jsonl")
    if hi_path.exists():
        his = [json.loads(l) for l in hi_path.open() if l.strip()]
        detected = sum(1 for h in his if h.get('llm_detected'))
        print(f"\n4. STATE & EVENT RECOVERY")
        print(f"   Total HIs processed: {len(his)}")
        print(f"   LLM auto-detected: {detected}/{len(his)}")
        print(f"   Manually added: {len(his) - detected}/{len(his)}")

        print(f"\n   State transitions:")
        for h in his:
            d = "Y" if h['llm_detected'] else "N"
            print(f"     [{d}] {h['hi_id']}: {h['pre_state']} -> {h['post_state']}  ({h['trigger_message']})")

    # 5. Test cases
    tc_path = Path("data/processed/test_cases.jsonl")
    if tc_path.exists():
        tcs = [json.loads(l) for l in tc_path.open() if l.strip()]
        sev = Counter(tc.get('severity', '?') for tc in tcs)
        typ = Counter(tc.get('test_type', '?') for tc in tcs)
        print(f"\n5. TEST CASE GENERATION")
        print(f"   Total test cases: {len(tcs)}")
        print(f"   From HIs: {len(his)}")
        print(f"   Avg per HI: {len(tcs)/len(his):.1f}")
        print(f"   By severity: {dict(sev)}")
        print(f"   By type: {dict(typ)}")

    # 6. Pipeline summary
    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"   PDF spec -> {len(paras)} paragraphs -> {len(gold_rows)} content")
    print(f"   -> {len(yes_rows)} hazard indicators -> {len(his)} structured HIs")
    print(f"   -> {len(tcs)} test cases")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()