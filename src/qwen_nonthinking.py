"""
Qwen 3.5 397B - Thinking DISABLED
===================================
Server must be started with: --chat-template-kwargs '{"enable_thinking":false}'
Output: qwen_nothinking_predictions.jsonl
"""
from pathlib import Path
import json
import re
import pandas as pd
import requests
import time

GOLD_PATH = Path("data/processed/annotation_sample.csv")
OUT_PATH = Path("data/processed/qwen_nothinking_predictions.jsonl")

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5"

ALLOWED_TYPES = [
    "ordering_violation", "state_violation", "exception_handling",
    "context_reset", "credential_handling", "other",
]

STRONG_RISKY = [
    "abort", "discard", "clear", "invalidate", "reset",
    "delete", "wipe", "flush"
]
MEDIUM_RISKY = [
    "release", "deactivate", "deactivation", "detach", "terminate",
    "reject", "remove", "revoke", "cancel", "stop", "drop"
]
CONDITION_TRIGGERS = [
    "if ", "unless ", "when ", "before ", "upon ", "on receipt",
    "on reception", "after receiving", "in case", "in the event",
    "initiates", "is initiated"
]
ABNORMAL_TRIGGERS = [
    "cannot", "failed", "failure", "not accepted", "invalid", "mismatch",
    "reject", "rejected", "timer expiry", "timeout", "error", "abnormal",
    "no response", "not available", "missing", "unexpected"
]
PURE_BENIGN_MARKERS = [
    "is defined as", "refers to", "for the purposes of",
    "in this specification", "is used to describe", "abbreviation for",
]


def prefilter_maybe_hazard(paragraph: str) -> bool:
    t = paragraph.lower()
    if t.startswith("this message is sent by"):
        return False
    DESCRIPTION_STARTERS = [
        "the state ", "the substate ", "the purpose of the ",
        "this procedure is used", "is used to "
    ]
    if any(t.startswith(s) for s in DESCRIPTION_STARTERS):
        if not any(v in t for v in STRONG_RISKY):
            return False
    if any(b in t for b in PURE_BENIGN_MARKERS):
        if not any(k in t for k in STRONG_RISKY + MEDIUM_RISKY):
            return False
    has_cond = any(k in t for k in CONDITION_TRIGGERS)
    has_abnormal = any(k in t for k in ABNORMAL_TRIGGERS)
    has_strong = any(k in t for k in STRONG_RISKY)
    has_medium = any(k in t for k in MEDIUM_RISKY)
    if (has_cond or has_abnormal) and (has_strong or has_medium):
        return True
    if has_strong:
        return True
    return False


def build_classifier_messages(paragraph: str):
    system = (
        "You are a strict LTE protocol security analyst. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\"} or {\"is_hazard\":\"no\"}. "
        "No extra text."
    )
    user = f"""
Task: Does this LTE specification paragraph contain a HAZARD INDICATOR?

A hazard indicator is a statement that links a trigger event to a DESTRUCTIVE or
SECURITY-RELEVANT operation — specifically one that causes loss of context, credentials,
or connectivity.

=== RETURN "yes" ONLY if the paragraph describes: ===
1. Bearer context being DEACTIVATED or DELETED in response to a message/event
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials (GUTI, KSI, TAI list) being DELETED or INVALIDATED
4. Context being RESET or RELEASED due to a reject or failure
5. State transition to EMM-DEREGISTERED triggered by an error condition

=== RETURN "no" for: ===
1. Message format tables ("This message is sent by...", IEI tables)
2. State definitions ("EMM-NULL: The EPS capability is disabled")
3. Normal procedural steps ("The UE shall send an ATTACH REQUEST")
4. A REJECT message being SENT as a response (sending a reject is normal, not hazardous)
5. General descriptions of procedure purposes
6. Timer definitions, abbreviation lists, scope descriptions
7. Editor's notes

=== Examples: ===

HAZARD (yes):
"Upon receipt of an AUTHENTICATION REJECT message, the mobile station shall set the
update status to EU3, delete the stored GUTI, TAI list, and KSI."
Reason: Credential deletion triggered by a specific message

NOT A HAZARD (no):
"If the bearer resource allocation requested cannot be accepted by the network, the
MME shall send a BEARER RESOURCE ALLOCATION REJECT message to the UE."
Reason: Just sending a reject message, no destructive action on context/credentials

NOT A HAZARD (no):
"The detach procedure is used by the UE to inform the network that it does not want
to access the EPS any longer."
Reason: General description of purpose, no specific destructive operation

Paragraph to classify:
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def build_extractor_messages(paragraph: str):
    system = (
        "You are an LTE protocol security analyst. "
        "Extract hazard indicator details. Return ONLY valid JSON. No extra text."
    )
    user = f"""
Extract the hazard indicator from this LTE specification paragraph.

=== FEW-SHOT EXAMPLES ===

Example 1:
Paragraph: "Upon receipt of an AUTHENTICATION REJECT message, the mobile station shall
set the update status in the USIM to EU3 ROAMING NOT ALLOWED, delete from the USIM the
stored GUTI, TAI list, last visited registered TAI and KSI."
Output:
{{"condition": "Receipt of AUTHENTICATION REJECT message", "operation": "Set update status to EU3 ROAMING NOT ALLOWED; delete GUTI, TAI list, last visited registered TAI, and KSI from USIM", "state": "Any EMM state -> EMM-DEREGISTERED", "hazard_type": "credential_handling"}}

Example 2:
Paragraph: "When receiving the DETACH REQUEST message and the Detach type IE indicates
re-attach required, the UE shall deactivate the EPS bearer context(s) including the
EPS default bearer context locally."
Output:
{{"condition": "Receipt of DETACH REQUEST with Detach type = re-attach required", "operation": "Deactivate all EPS bearer contexts including default bearer locally (no peer-to-peer signalling)", "state": "EMM-REGISTERED -> EMM-DEREGISTERED", "hazard_type": "context_reset"}}

Example 3:
Paragraph: "Upon receipt of the ACTIVATE DEDICATED EPS BEARER CONTEXT REJECT message
in state BEARER CONTEXT ACTIVE PENDING, the MME shall enter the state BEARER CONTEXT
INACTIVE and abort the dedicated EPS bearer context activation procedure."
Output:
{{"condition": "Receipt of ACTIVATE DEDICATED EPS BEARER CONTEXT REJECT in state BEARER CONTEXT ACTIVE PENDING", "operation": "Enter BEARER CONTEXT INACTIVE; abort dedicated EPS bearer context activation procedure", "state": "BEARER CONTEXT ACTIVE PENDING -> BEARER CONTEXT INACTIVE", "hazard_type": "exception_handling"}}

=== FIELD DEFINITIONS ===
- condition: The specific trigger message or event. Copy the exact message name.
- operation: The destructive/risky action(s). List all actions separated by semicolons.
- state: Protocol state before -> after. Use exact state names from the spec. If unclear, use "".
- hazard_type: One of:
  - "context_reset" = bearer/EPS context deactivated, deleted, or released
  - "credential_handling" = GUTI/KSI/TAI list deleted or invalidated
  - "exception_handling" = procedure aborted due to rejection, collision, or failure
  - "state_violation" = unexpected state transition or state mismatch
  - "ordering_violation" = procedure collision, out-of-order message handling
  - "other" = does not fit above categories

=== NOW EXTRACT FROM THIS PARAGRAPH ===
\"\"\"{paragraph}\"\"\"

Return ONLY the JSON object. If you cannot identify a field, use empty string "".
"""
    return system, user


def postfilter_valid_extraction(operation: str, paragraph: str) -> bool:
    op_lower = operation.lower()
    para_lower = paragraph.lower()
    has_risky_in_op = any(v in op_lower for v in STRONG_RISKY + MEDIUM_RISKY)
    has_risky_in_para = any(v in para_lower for v in STRONG_RISKY + MEDIUM_RISKY)
    if not (has_risky_in_op or has_risky_in_para):
        return False
    SEND_ONLY_PATTERNS = ["send a", "send the", "shall send", "return a", "return the"]
    op_is_send_only = any(p in op_lower for p in SEND_ONLY_PATTERNS)
    if op_is_send_only and not any(v in op_lower for v in STRONG_RISKY):
        return False
    return True


def call_llm(system: str, user: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(LLAMA_CPP_URL, json=payload, timeout=900)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()

    # Strip thinking tags if present (shouldn't be with thinking=false, but just in case)
    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return {}
        return {}


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


def main():
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold file not found: {GOLD_PATH}")

    # Verify server config
    print("=" * 60)
    print("QWEN 3.5 397B — THINKING DISABLED")
    print("=" * 60)
    print("IMPORTANT: Ensure server was started with:")
    print("  --chat-template-kwargs '{\"enable_thinking\":false}'")
    print()

    df = pd.read_csv(GOLD_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(OUT_PATH)

    stats = {"prefilter_blocked": 0, "llm_rejected": 0,
             "postfilter_blocked": 0, "accepted": 0}

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pid = int(row["id"])
            if pid in done_ids:
                continue
            paragraph = str(row["paragraph"])

            if not prefilter_maybe_hazard(paragraph):
                stats["prefilter_blocked"] += 1
                print(f"ID {pid}: prefilter BLOCKED")
                pred = {"is_hazard": "no", "condition": "", "operation": "",
                        "state": "", "hazard_type": ""}
            else:
                print(f"ID {pid}: sending to LLM...")
                sysA, usrA = build_classifier_messages(paragraph)
                cls = call_llm(sysA, usrA)
                is_hazard = str(cls.get("is_hazard", "no")).strip().lower()
                if is_hazard not in ["yes", "no"]:
                    is_hazard = "no"

                if is_hazard == "no":
                    stats["llm_rejected"] += 1
                    pred = {"is_hazard": "no", "condition": "", "operation": "",
                            "state": "", "hazard_type": ""}
                else:
                    sysB, usrB = build_extractor_messages(paragraph)
                    ext = call_llm(sysB, usrB)
                    operation = str(ext.get("operation", "")).strip()

                    if not postfilter_valid_extraction(operation, paragraph):
                        stats["postfilter_blocked"] += 1
                        pred = {"is_hazard": "no", "condition": "", "operation": "",
                                "state": "", "hazard_type": ""}
                    else:
                        stats["accepted"] += 1
                        pred = {
                            "is_hazard": "yes",
                            "condition": str(ext.get("condition", "")).strip(),
                            "operation": operation,
                            "state": str(ext.get("state", "")).strip(),
                            "hazard_type": str(ext.get("hazard_type", "")).strip(),
                        }
                        if pred["hazard_type"] not in ALLOWED_TYPES:
                            pred["hazard_type"] = "other"

            record = {
                "id": pid,
                "paragraph": paragraph,
                "qwen_nothinking_is_hazard": pred["is_hazard"],
                "qwen_nothinking_condition": pred["condition"],
                "qwen_nothinking_operation": pred["operation"],
                "qwen_nothinking_state": pred["state"],
                "qwen_nothinking_hazard_type": pred["hazard_type"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(0.1)

    print(f"\nDone. Filter stats: {stats}")
    print(f"Predictions saved to {OUT_PATH}")


if __name__ == "__main__":
    main()