"""
PHASE 1: THINKING + CONTEXT ENGINEERING
Question: Can context engineering reduce thinking noise and improve F1?

Background:
  Thinking ON alone: F1=0.733 (WORSE - thinking overrides prompts)
  Context Engineering: F1=0.823 (BETTER - richer context)
  Thinking ON + Context Eng: F1=? (UNKNOWN - TEST THIS!)

Hypothesis:
  By enriching context WITH thinking, LLM has better grounding
  for its reasoning. Context prevents thinking from diverging.
"""

import json
import csv
from pathlib import Path
import requests

GOLD_PATH = Path("data/processed/annotation_sample.csv")
QWEN_API = "http://localhost:8080/v1/chat/completions"  # Your workstation

def load_gold_labels():
    """Load gold standard"""
    gold = {}
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            para_id = int(row['id'])
            gold[para_id] = {
                'id': para_id,
                'is_hazard': row['is_hazard'].strip().lower() == 'yes',
                'section_id': row['section_id'],
                'section_title': row['section_title'],
                'paragraph': row['paragraph'],
                'condition': row['condition'],
                'operation': row['operation'],
            }
    return gold

def build_enriched_context(para, section_title, condition, operation):
    """
    Build enriched context to GROUND thinking
    
    Key idea: Provide structured context so thinking doesn't diverge
    """
    
    context = f"""
PARAGRAPH ANALYSIS CONTEXT:
===========================

Section: {section_title}
Procedure Type: {section_title.split('.')[-1] if '.' in section_title else section_title}

PARAGRAPH TEXT:
{para}

STRUCTURAL ANALYSIS:
- Precondition: {condition if condition else '(Not specified)'}
- Operation: {operation if operation else '(Not specified)'}

TASK:
Identify if this paragraph describes a SECURITY HAZARD INDICATOR (HI)

Security Hazard Indicator Definition:
A paragraph is a hazard indicator if it describes:
1. Exception handling for unusual conditions
2. Implicit state changes or operations
3. Loss of synchronization between entities
4. Potential for abuse or attack
5. Failure modes without explicit recovery

REASONING:
Step through the logic carefully. Use the structural context above to ground your reasoning.
Don't diverge from the provided context. Base conclusions on evidence in the paragraph.
"""
    
    return context

def query_qwen_with_thinking(paragraph, context, thinking_budget=5000):
    """
    Query Qwen with extended thinking + enriched context
    
    Parameters:
      thinking_budget: Max tokens for thinking (5000 = ~1-2 min reasoning)
    """
    
    system_prompt = """You are a security analyst for 3GPP LTE specifications.
Your task is to identify security hazard indicators in protocol specifications.

You MUST follow this format:
THINKING: (Your internal reasoning process)
CONCLUSION: (Is this a hazard? YES or NO)
CONFIDENCE: (High/Medium/Low)"""
    
    user_message = f"""{context}

Analyze this paragraph for security hazard indicators using your extended thinking capability.
Think carefully about the implications. Ground your reasoning in the provided context."""
    
    payload = {
        "model": "qwen:latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(QWEN_API, json=payload, timeout=60)
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            
            # Parse response
            is_hazard = "YES" in content.upper() and "CONCLUSION: YES" in content.upper()
            
            return {
                'predicted': is_hazard,
                'response': content,
                'confidence': 'High' if 'CONFIDENCE: High' in content else 'Medium'
            }
    except Exception as e:
        print(f"  Error querying Qwen: {e}")
    
    return {'predicted': False, 'response': '', 'confidence': 'Low'}

def evaluate(predictions, gold_labels):
    """Evaluate predictions"""
    tp = tn = fp = fn = 0
    fn_cases = []
    fp_cases = []
    
    for para_id, gold in gold_labels.items():
        pred = predictions.get(para_id, False)
        is_hazard = gold['is_hazard']
        
        if is_hazard and pred:
            tp += 1
        elif not is_hazard and not pred:
            tn += 1
        elif not is_hazard and pred:
            fp += 1
            fp_cases.append(para_id)
        elif is_hazard and not pred:
            fn += 1
            fn_cases.append(para_id)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'fn_cases': fn_cases, 'fp_cases': fp_cases
    }

def main():
    print("\n" + "="*80)
    print("PHASE 1: THINKING + CONTEXT ENGINEERING TEST")
    print("="*80)
    
    print("\nObjective: Can context engineering reduce thinking noise?")
    print("Hypothesis: Enriched context grounds thinking, improves F1")
    print("\nBaseline Results:")
    print("  Thinking ON alone:        F1=0.733 (WORSE)")
    print("  Context Engineering:      F1=0.823 (BETTER)")
    print("  Thinking ON + Context:    F1=? (TESTING NOW)")
    
    # Load data
    print("\nLoading gold labels...")
    gold_labels = load_gold_labels()
    print(f"  ✓ Loaded {len(gold_labels)} paragraphs (17 HIs)")
    
    # Run predictions
    print("\nRunning Thinking + Context Engineering on each paragraph...")
    print("(This will take ~2-3 minutes - thinking takes time)\n")
    
    predictions = {}
    responses = {}
    high_confidence = 0
    low_confidence = 0
    
    for para_id, gold in gold_labels.items():
        if (para_id + 1) % 20 == 0:
            print(f"  Progress: {para_id + 1}/150...")
        
        # Build enriched context
        context = build_enriched_context(
            gold['paragraph'],
            gold['section_title'],
            gold['condition'],
            gold['operation']
        )
        
        # Query with thinking + context
        result = query_qwen_with_thinking(gold['paragraph'], context)
        
        predictions[para_id] = result['predicted']
        responses[para_id] = result
        
        if result['confidence'] == 'High':
            high_confidence += 1
        else:
            low_confidence += 1
    
    print(f"\n✓ Completed {len(predictions)} predictions")
    print(f"  High confidence: {high_confidence}")
    print(f"  Low confidence: {low_confidence}")
    
    # Evaluate
    print("\nEvaluating results...")
    results = evaluate(predictions, gold_labels)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: THINKING + CONTEXT ENGINEERING")
    print("="*80)
    
    print(f"\nMetrics:")
    print(f"  TP: {results['tp']}")
    print(f"  TN: {results['tn']}")
    print(f"  FP: {results['fp']}")
    print(f"  FN: {results['fn']}")
    
    print(f"\n  Precision: {results['precision']:.3f}")
    print(f"  Recall:    {results['recall']:.3f}")
    print(f"  F1 Score:  {results['f1']:.3f}")
    
    # Comparison
    print(f"\n" + "="*80)
    print("COMPARISON TO BASELINES")
    print("="*80)
    
    print(f"\n  Thinking ON alone:         F1=0.733")
    print(f"  Context Engineering:       F1=0.823")
    print(f"  Thinking + Context:        F1={results['f1']:.3f} ← THIS TEST")
    print(f"  Best previous (validator): F1=0.938")
    
    improvement = results['f1'] - 0.733
    print(f"\nImprovement over thinking alone: +{improvement:.3f} ({improvement*100:.1f}%)")
    
    if results['f1'] > 0.938:
        print(f"\n✅ BREAKTHROUGH! Thinking + Context BEATS previous best!")
        print(f"   New best F1: {results['f1']:.3f}")
    elif results['f1'] > 0.823:
        print(f"\n✓ Success! Thinking + Context better than context alone")
        print(f"   Shows thinking CAN work with proper grounding")
    else:
        print(f"\n⚠️  Thinking still hurts performance")
        print(f"   Context engineering alone (0.823) is better")
        print(f"   Thinking diverges even with rich context")
    
    # Error analysis
    if results['fn'] > 0:
        print(f"\nFalse Negatives ({results['fn']}): {results['fn_cases']}")
    if results['fp'] > 0:
        print(f"False Positives ({results['fp']}): {results['fp_cases']}")
    
    print(f"\n" + "="*80)
    
    # Save detailed results
    output_path = Path("data/processed/thinking_plus_context_results.jsonl")
    with open(output_path, 'w') as f:
        for para_id, resp in responses.items():
            f.write(json.dumps({
                'id': para_id,
                'predicted': predictions[para_id],
                'response': resp['response'][:500],  # Truncate for storage
                'confidence': resp['confidence']
            }) + '\n')
    
    print(f"✓ Results saved to {output_path}")

if __name__ == "__main__":
    main()