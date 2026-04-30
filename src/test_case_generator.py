from pathlib import Path
import json
import requests
import csv
import time

HI_PATH = Path("data/processed/hazard_indicators_structured.jsonl")
OUT_PATH = Path("data/processed/test_cases.jsonl")
SUMMARY_PATH = Path("data/processed/test_case_summary.csv")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


def build_testgen_prompt(hi):
    system = (
        "You are a telecom test engineer. "
        "Return ONLY a JSON array of test case objects. No extra text."
    )
    user = f"""
Generate 2 test cases for this LTE protocol hazard.

Hazard:
- Trigger: {hi['trigger_message']} ({hi.get('trigger_direction', 'unknown')})
- Entity: {hi.get('entity', 'unknown')}
- State: {hi['pre_state']} -> {hi['post_state']}
- Operation: {hi['gold_operation'][:200]}
- Type: {hi['gold_hazard_type']}

Generate:
1. NORMAL test: verify the hazard occurs as specified
2. EDGE CASE test: trigger arrives in unexpected state or timing

Return a JSON array:
[
  {{
    "name": "short test name",
    "description": "what this verifies",
    "preconditions": ["precondition 1", "precondition 2"],
    "steps": ["step 1", "step 2", "step 3"],
    "expected_state": "final state",
    "expected_operations": ["expected action 1", "expected action 2"],
    "severity": "high" or "medium" or "low",
    "test_type": "negative" or "boundary"
  }}
]
"""
    return system, user


def call_ollama(system, user):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 600}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=900)
    r.raise_for_status()
    text = r.json()["message"]["content"].strip()

    # Try parsing as JSON array
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except:
        pass

    # Try finding array in text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except:
            pass

    # Try single object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return [json.loads(text[start:end+1])]
        except:
            pass

    return []


def main():
    # Load structured HIs
    his = []
    with HI_PATH.open() as f:
        for line in f:
            if line.strip():
                his.append(json.loads(line))

    print(f"Loaded {len(his)} structured hazard indicators.\n")

    all_test_cases = []
    tc_counter = 0

    for hi in his:
        print(f"Generating test cases for {hi['hi_id']} ({hi['trigger_message']})...")
        system, user = build_testgen_prompt(hi)
        raw_cases = call_ollama(system, user)

        if not raw_cases:
            print(f"  WARNING: No test cases generated, using fallback")
            raw_cases = [{
                "name": f"Verify {hi['trigger_message']} handling",
                "description": f"Test that {hi['gold_operation'][:100]}",
                "preconditions": [f"UE/MME in {hi['pre_state']}"],
                "steps": [f"Send {hi['trigger_message']}", "Observe state transition"],
                "expected_state": hi['post_state'],
                "expected_operations": hi.get('operations', []),
                "severity": "high" if hi['gold_hazard_type'] == 'credential_handling' else "medium",
                "test_type": "negative"
            }]

        for raw in raw_cases:
            tc_counter += 1
            tc = {
                "tc_id": f"TC-{hi['hi_id']}-{tc_counter:02d}",
                "hi_id": hi['hi_id'],
                "name": raw.get("name", ""),
                "description": raw.get("description", ""),
                "preconditions": raw.get("preconditions", []),
                "steps": raw.get("steps", []),
                "expected_state": raw.get("expected_state", ""),
                "expected_operations": raw.get("expected_operations", []),
                "severity": raw.get("severity", "medium"),
                "test_type": raw.get("test_type", "negative"),
            }
            all_test_cases.append(tc)
            print(f"  {tc['tc_id']}: {tc['name'][:60]}")

        time.sleep(0.1)

    # Save test cases JSONL
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for tc in all_test_cases:
            f.write(json.dumps(tc, ensure_ascii=False) + "\n")

    # Save summary CSV
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tc_id", "hi_id", "name", "severity", "test_type",
                         "expected_state", "num_steps"])
        for tc in all_test_cases:
            writer.writerow([
                tc["tc_id"], tc["hi_id"], tc["name"][:60], tc["severity"],
                tc["test_type"], tc["expected_state"], len(tc["steps"])
            ])

    # Print stats
    from collections import Counter
    severity_dist = Counter(tc["severity"] for tc in all_test_cases)
    type_dist = Counter(tc["test_type"] for tc in all_test_cases)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_test_cases)} test cases from {len(his)} HIs")
    print(f"Saved to {OUT_PATH}")
    print(f"Summary saved to {SUMMARY_PATH}")
    print(f"\nSeverity: {dict(severity_dist)}")
    print(f"Type: {dict(type_dist)}")


if __name__ == "__main__":
    main()