import json
from pathlib import Path

PATH = Path("data/processed/hazard_indicators_structured.jsonl")

# Load
records = []
with PATH.open() as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Corrections
fixes = {
    "HI-001": {"pre_state": "EMM-REGISTERED", "post_state": "EMM-REGISTERED", "trigger_message": "NAS signalling message (integrity check fail)"},
    "HI-002": {"pre_state": "EMM-COMMON-PROCEDURE-INITIATED", "post_state": "EMM-COMMON-PROCEDURE-INITIATED", "trigger_message": "AUTHENTICATION FAILURE"},
    "HI-003": {"pre_state": "EMM-REGISTERED", "post_state": "EMM-DEREGISTERED", "trigger_message": "AUTHENTICATION REJECT"},
    "HI-004": {"pre_state": "EMM-REGISTERED", "post_state": "EMM-DEREGISTERED", "trigger_message": "AUTHENTICATION FAILURE"},
    "HI-006": {"trigger_message": "DETACH REQUEST"},
    "HI-007": {"pre_state": "EMM-REGISTERED", "trigger_message": "DETACH REQUEST"},
    "HI-008": {"trigger_message": "DETACH REQUEST"},
    "HI-009": {"trigger_message": "DETACH REQUEST"},
    "HI-010": {"pre_state": "EMM-DEREGISTERED-INITIATED", "post_state": "EMM-DEREGISTERED", "trigger_message": "DETACH ACCEPT"},
    "HI-011": {"pre_state": "EMM-TRACKING-AREA-UPDATING-INITIATED", "post_state": "EMM-REGISTERED", "trigger_message": "TRACKING AREA UPDATE ACCEPT"},
    "HI-013": {"pre_state": "BEARER CONTEXT ACTIVE PENDING", "post_state": "BEARER CONTEXT INACTIVE", "trigger_message": "ACTIVATE DEDICATED EPS BEARER CONTEXT REJECT"},
    "HI-014": {"pre_state": "BEARER CONTEXT MODIFY PENDING", "post_state": "BEARER CONTEXT ACTIVE", "trigger_message": "SESSION MANAGEMENT CONFIGURATION REJECT"},
    "HI-015": {"pre_state": "BEARER CONTEXT ACTIVE", "post_state": "BEARER CONTEXT INACTIVE", "trigger_message": "SESSION MANAGEMENT CONFIGURATION REQUEST"},
    "HI-016": {"pre_state": "BEARER CONTEXT ACTIVE", "post_state": "BEARER CONTEXT INACTIVE", "trigger_message": "SESSION MANAGEMENT CONFIGURATION REQUEST"},
    "HI-017": {"pre_state": "BEARER CONTEXT ACTIVE", "post_state": "BEARER CONTEXT INACTIVE", "trigger_message": "PDN DISCONNECT REQUEST"},
}

for rec in records:
    hi_id = rec["hi_id"]
    if hi_id in fixes:
        for key, val in fixes[hi_id].items():
            rec[key] = val

# Save
with PATH.open("w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Print summary
print("CORRECTED STATE TRANSITION SUMMARY:")
print("=" * 60)
for rec in records:
    detected = "Y" if rec["llm_detected"] else "N"
    print(f"  [{detected}] {rec['hi_id']}: {rec['pre_state']} -> {rec['post_state']}  (on {rec['trigger_message']})")

print(f"\nFixed {len(fixes)} records.")