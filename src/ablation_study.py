"""
PHASE 1: ABLATION STUDY
Systematically remove each component and measure its contribution to F1

Expected progression:
  1. Vanilla (single model):        F1 ≈ 0.786
  2. + Consensus (multi-model):     F1 ≈ 0.903 (+0.117)
  3. + Context Engineering:         F1 ≈ 0.823 (may decrease due to FPs)
  4. + Validator:                   F1 ≈ 0.938 (+0.115 from validator)

This script compares prediction files from each stage
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OUTPUT_PATH = Path("data/processed/phase1_ablation_results.csv")

# Map each stage to the prediction file(s) you have
ABLATION_STAGES = [
    {
        "name": "Vanilla (Llama 3.2 3B)",
        "files": ["llama_predictions.jsonl", "vanilla_predictions.jsonl"],
        "description": "Single model, no consensus"
    },
    {
        "name": "Consensus (Llama + Qwen)",
        "files": ["consensus_predictions.jsonl"],
        "description": "Multi-model voting, no context or validator"
    },
    {
        "name": "+ Context Engineering",
        "files": ["context_predictions.jsonl", "agentic_step3_context_predictions.jsonl"],
        "description": "Consensus + context enrichment (may have more FPs)"
    },
    {
        "name": "+ Skeptical Validator",
        "files": ["validator_predictions.jsonl", "agentic_step5_combined_predictions.jsonl"],
        "description": "Consensus + context + validator (best pipeline)"
    },
    {
        "name": "+ Attacker Validator",
        "files": ["agentic_step5b_predictions.jsonl"],
        "description": "Alternative validator (attacker perspective)"
    },
]

def load_gold_labels():
    """Load ground truth annotations"""
    gold = {}
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            para_id = int(row['id'])
            gold[para_id] = {
                'is_hazard': row['is_hazard'].strip().lower() == 'yes',
            }
    return gold

def find_prediction_file(file_options):
    """Try to find first available prediction file from options"""
    for filename in file_options:
        path = Path("data/processed") / filename
        if path.exists():
            return path
    return None

def load_predictions(file_path):
    """Load predictions from JSON file"""
    predictions = {}
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    para_id = int(obj.get('id', obj.get('paragraph_id', -1)))
                    if para_id >= 0:
                        # Handle different prediction formats
                        pred = obj.get('consensus_is_hazard') or obj.get('is_hazard') or obj.get('predicted')
                        
                        # Normalize to boolean
                        if isinstance(pred, str):
                            predicted = pred.lower() in ['yes', 'true', '1']
                        else:
                            predicted = bool(pred)
                        
                        predictions[para_id] = predicted
        return predictions if predictions else None
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

def calculate_metrics(predictions, gold_labels):
    """Calculate P, R, F1 for a prediction set"""
    tp = fn = fp = 0
    
    for para_id, gold in gold_labels.items():
        pred = predictions.get(para_id)
        
        if pred is None:
            continue
        
        if gold and pred:
            tp += 1
        elif gold and not pred:
            fn += 1
        elif not gold and pred:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def run_ablation_study():
    """Run ablation study across all stages"""
    
    print("\n" + "="*80)
    print("PHASE 1: ABLATION STUDY")
    print("="*80)
    print("\nLoading gold labels...")
    gold_labels = load_gold_labels()
    print(f"  Loaded {len(gold_labels)} annotations (17 HIs expected)")
    
    results = []
    baseline_f1 = None
    
    print("\nTesting each ablation stage...\n")
    
    for stage in ABLATION_STAGES:
        print(f"Stage: {stage['name']}")
        print(f"  Description: {stage['description']}")
        
        # Find prediction file
        pred_file = find_prediction_file(stage['files'])
        
        if not pred_file:
            print(f"  ❌ No prediction file found. Tried: {stage['files']}")
            print()
            continue
        
        print(f"  File: {pred_file.name}")
        
        # Load and evaluate
        predictions = load_predictions(pred_file)
        
        if not predictions:
            print(f"  ❌ Could not load predictions")
            print()
            continue
        
        metrics = calculate_metrics(predictions, gold_labels)
        
        # Calculate delta from baseline
        if baseline_f1 is None:
            baseline_f1 = metrics['f1']
            delta = 0
        else:
            delta = metrics['f1'] - baseline_f1
        
        # Print results
        print(f"  TP={metrics['tp']}, FN={metrics['fn']}, FP={metrics['fp']}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}", end="")
        
        if delta != 0:
            sign = "+" if delta > 0 else ""
            print(f"  {sign}{delta:+.3f} from baseline")
        else:
            print()
        print()
        
        results.append({
            'stage': stage['name'],
            'description': stage['description'],
            'file': pred_file.name,
            'tp': metrics['tp'],
            'fn': metrics['fn'],
            'fp': metrics['fp'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'delta_f1': delta
        })
    
    # Print ablation table
    print("="*80)
    print("ABLATION TABLE")
    print("="*80)
    print()
    print(f"{'Stage':<35} {'F1':<8} {'Prec':<8} {'Recall':<8} {'ΔF1':<10}")
    print("-"*80)
    
    for r in results:
        delta_str = f"{r['delta_f1']:+.3f}" if r['delta_f1'] != 0 else "—"
        print(f"{r['stage']:<35} {r['f1']:.3f}   {r['precision']:.3f}   {r['recall']:.3f}   {delta_str:<10}")
    
    print()
    
    # Analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    
    if len(results) >= 2:
        consensus_delta = results[1]['delta_f1'] if len(results) > 1 else 0
        print(f"\n1. Consensus Impact:")
        print(f"   Consensus adds {consensus_delta:+.3f} to F1 (baseline → multi-model voting)")
        
    if len(results) >= 3:
        context_delta = results[2]['delta_f1'] - results[1]['delta_f1'] if len(results) > 2 else 0
        print(f"\n2. Context Engineering Impact:")
        print(f"   Context adds {context_delta:+.3f} to F1")
        if results[2]['fp'] > results[1]['fp']:
            print(f"   ⚠️  But introduces {results[2]['fp'] - results[1]['fp']} false positives")
    
    if len(results) >= 4:
        validator_delta = results[3]['delta_f1'] - results[2]['delta_f1'] if len(results) > 3 else 0
        print(f"\n3. Validator Impact:")
        print(f"   Validator adds {validator_delta:+.3f} to F1")
        print(f"   Eliminates {results[2]['fp'] - results[3]['fp']} false positives")
    
    print(f"\n4. Overall Progression:")
    if results:
        print(f"   Baseline F1:  {results[0]['f1']:.3f}")
        if len(results) > 1:
            print(f"   Best F1:      {max(r['f1'] for r in results):.3f}")
            best_stage = max(results, key=lambda x: x['f1'])
            print(f"   Best stage:   {best_stage['stage']}")
    
    # Save to CSV
    save_ablation_csv(results)
    
    print(f"\n✓ Ablation results saved to {OUTPUT_PATH}")
    print("="*80)

def save_ablation_csv(results):
    """Save ablation results to CSV for thesis"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'stage', 'description', 'file', 'tp', 'fn', 'fp', 
            'precision', 'recall', 'f1', 'delta_f1'
        ])
        writer.writeheader()
        writer.writerows(results)

def main():
    run_ablation_study()

if __name__ == "__main__":
    main()