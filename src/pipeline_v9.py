"""
FULL PIPELINE on TS 24.301 V9.0.0 (675 paragraphs, Bookworm scope)
====================================================================
Runs the best pipeline (Consensus + Validated Context Engineering)
on the expanded dataset. Since this dataset is unannotated, the script
pre-annotates everything — the user then manually reviews flagged
paragraphs to build the gold standard.

Steps:
1. Llama 3.2 3B classification (via Ollama)
2. Qwen 397B classification (via llama.cpp, thinking OFF)
3. Consensus union (Llama OR Qwen)
4. Context engineering on consensus-NO with risk score >= 3
5. Validation of context-rescued paragraphs
6. Output: pre-annotated CSV for manual review

Usage:
  python src\pipeline_v9.py llama       # Step 1: Llama predictions
  python src\pipeline_v9.py qwen        # Step 2: Qwen predictions  
  python src\pipeline_v9.py consensus   # Step 3: Build consensus
  python src\pipeline_v9.py context     # Step 4: Context engineering
  python src\pipeline_v9.py validate    # Step 5: Validate rescues
  python src\pipeline_v9.py report      # Step 6: Generate annotated CSV
"""
from pathlib import Path
import json
import re
import csv
import requests
import time
import sys

# ── Paths ──
DATA_DIR = Path("data/processed")
PARAGRAPHS_PATH = DATA_DIR / "TS_24_301_v9_bookworm.jsonl"

LLAMA_PREDS = DATA_DIR / "v9_llama_predictions.jsonl"
QWEN_PREDS = DATA_DIR / "v9_qwen_predictions.jsonl"
CONSENSUS_PREDS = DATA_DIR / "v9_consensus_predictions.jsonl"
CONTEXT_PREDS = DATA_DIR / "v9_context_predictions.jsonl"
VALIDATED_PREDS = DATA_DIR / "v9_validated_predictions.jsonl"
ANNOTATED_CSV = DATA_DIR / "v9_pre_annotated.csv"

# ── API endpoints ──
OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"

# ── Keyword lists ──
STRONG_RISKY = ["abort", "discard", "clear", "invalidate", "reset", "delete", "wipe", "flush"]
MEDIUM_RISKY = ["release", "deactivate", "deactivation", "detach", "terminate",
                "reject", "remove", "revoke", "cancel", "stop", "drop"]
CONDITION_TRIGGERS = ["if ", "unless ", "when ", "before ", "upon ", "on receipt",
                      "on reception", "after receiving", "in case", "in the event"]
ABNORMAL_TRIGGERS = ["cannot", "failed", "failure", "not accepted", "invalid", "mismatch",
                     "reject", "rejected", "timer expiry", "timeout", "error", "abnormal"]
HIGH_RISK_SECTIONS = ["abnormal", "not accepted", "reject", "failure", "error",
                      "integrity checking", "detach", "deregistration", "security mode"]


def load_paragraphs():
    records = []
    with PARAGRAPHS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_done_ids(path):
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        done.add(int(obj["id"]))
                    except:
                        pass
    return done


def load_predictions(path):
    preds = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    preds[int(obj["id"])] = obj
    return preds


def prefilter_maybe_hazard(paragraph):
    t = paragraph.lower()
    if t.startswith("this message is sent by"):
        return False
    desc_starters = ["the state ", "the substate ", "the purpose of the ",
                     "this procedure is used", "is used to "]
    if any(t.startswith(s) for s in desc_starters):
        if not any(v in t for v in STRONG_RISKY):
            return False
    benign = ["is defined as", "refers to", "for the purposes of",
              "in this specification", "abbreviation for"]
    if any(b in t for b in benign):
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


def build_classifier_prompt(paragraph):
    system = (
        "You are a strict LTE protocol security analyst. "
        "Return ONLY JSON: {\"is_hazard\":\"yes\"} or {\"is_hazard\":\"no\"}. "
        "No extra text."
    )
    user = f"""
Task: Does this LTE specification paragraph contain a HAZARD INDICATOR?

A hazard indicator links a trigger event to a DESTRUCTIVE or SECURITY-RELEVANT
operation — one that causes loss of context, credentials, or connectivity.

=== RETURN "yes" ONLY if the paragraph describes: ===
1. Bearer context being DEACTIVATED or DELETED in response to a message/event
2. Procedure being ABORTED due to collision, rejection, or failure
3. Security credentials (GUTI, KSI, TAI list) being DELETED or INVALIDATED
4. Context being RESET or RELEASED due to a reject or failure
5. State transition to EMM-DEREGISTERED triggered by an error condition
6. Messages being DISCARDED due to failed integrity/security checks

=== RETURN "no" for: ===
1. Message format tables
2. State definitions without trigger-action pairs
3. Normal procedural steps ("The UE shall send an ATTACH REQUEST")
4. A REJECT message being SENT (sending is not destructive)
5. General descriptions of procedure purposes
6. Timer definitions, abbreviation lists, scope descriptions

=== Examples: ===
HAZARD (yes):
"Upon receipt of an AUTHENTICATION REJECT message, the mobile station shall set the
update status to EU3, delete the stored GUTI, TAI list, and KSI."

NOT A HAZARD (no):
"If the bearer resource allocation cannot be accepted, the MME shall send a BEARER
RESOURCE ALLOCATION REJECT message to the UE."

NOT A HAZARD (no):
"The detach procedure is used by the UE to inform the network that it does not want
to access the EPS any longer."

Paragraph to classify:
\"\"\"{paragraph}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
    return system, user


def call_ollama(system, user):
    payload = {
        "model": "llama3.2:3b",
        "prompt": system + "\n\n" + user,
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    text = r.json()["response"].strip()
    return parse_json(text)


def call_qwen(system, user):
    payload = {
        "model": "qwen3.5",
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
    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()
    return parse_json(text)


def parse_json(text):
    try:
        return json.loads(text)
    except:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except:
                return {}
        return {}


def compute_risk_score(paragraph, section_title):
    t = paragraph.lower()
    score = 0
    if any(t.startswith(p) for p in ["this message is sent by", "the state ", "the substate ",
                                      "the purpose of the ", "is defined as"]):
        return 0
    for w in STRONG_RISKY:
        if w in t: score += 3
    for w in MEDIUM_RISKY:
        if w in t: score += 1
    if "shall" in t and any(v in t for v in STRONG_RISKY + MEDIUM_RISKY):
        score += 2
    if any(c in t for c in ["if ", "upon ", "when ", "on receipt"]):
        score += 2
    if "enter" in t and "state" in t:
        score += 2
    st = section_title.lower()
    if any(h in st for h in HIGH_RISK_SECTIONS):
        score += 3
    return score


# ═══════════════════════════════════════════════════════════════
# STEP 1: Llama predictions
# ═══════════════════════════════════════════════════════════════
def run_llama():
    records = load_paragraphs()
    done_ids = load_done_ids(LLAMA_PREDS)
    remaining = len(records) - len(done_ids)
    print(f"STEP 1: Llama 3.2 3B — {len(records)} paragraphs, {remaining} remaining")

    with LLAMA_PREDS.open("a", encoding="utf-8") as f:
        for rec in records:
            pid = rec["id"]
            if pid in done_ids:
                continue
            para = rec["paragraph"]
            if not prefilter_maybe_hazard(para):
                is_hazard = "no"
            else:
                sys_p, usr_p = build_classifier_prompt(para)
                result = call_ollama(sys_p, usr_p)
                is_hazard = str(result.get("is_hazard", "no")).lower()
                if is_hazard not in ["yes", "no"]:
                    is_hazard = "no"

            out = {"id": pid, "is_hazard": is_hazard}
            f.write(json.dumps(out) + "\n")
            f.flush()
            status = "YES" if is_hazard == "yes" else "no" if prefilter_maybe_hazard(para) else "skip"
            print(f"  ID {pid}: {status}")
            time.sleep(0.05)

    total_yes = sum(1 for p in load_predictions(LLAMA_PREDS).values() if p["is_hazard"] == "yes")
    print(f"\nLlama done: {total_yes} hazards detected out of {len(records)}")


# ═══════════════════════════════════════════════════════════════
# STEP 2: Qwen predictions
# ═══════════════════════════════════════════════════════════════
def run_qwen():
    records = load_paragraphs()
    done_ids = load_done_ids(QWEN_PREDS)
    remaining = len(records) - len(done_ids)
    print(f"STEP 2: Qwen 397B (thinking OFF) — {len(records)} paragraphs, {remaining} remaining")

    with QWEN_PREDS.open("a", encoding="utf-8") as f:
        for rec in records:
            pid = rec["id"]
            if pid in done_ids:
                continue
            para = rec["paragraph"]
            if not prefilter_maybe_hazard(para):
                is_hazard = "no"
            else:
                sys_p, usr_p = build_classifier_prompt(para)
                result = call_qwen(sys_p, usr_p)
                is_hazard = str(result.get("is_hazard", "no")).lower()
                if is_hazard not in ["yes", "no"]:
                    is_hazard = "no"

            out = {"id": pid, "is_hazard": is_hazard}
            f.write(json.dumps(out) + "\n")
            f.flush()
            status = "YES" if is_hazard == "yes" else "no" if prefilter_maybe_hazard(para) else "skip"
            print(f"  ID {pid}: {status}")
            time.sleep(0.05)

    total_yes = sum(1 for p in load_predictions(QWEN_PREDS).values() if p["is_hazard"] == "yes")
    print(f"\nQwen done: {total_yes} hazards detected out of {len(records)}")


# ═══════════════════════════════════════════════════════════════
# STEP 3: Consensus
# ═══════════════════════════════════════════════════════════════
def run_consensus():
    records = load_paragraphs()
    llama = load_predictions(LLAMA_PREDS)
    qwen = load_predictions(QWEN_PREDS)

    if not llama or not qwen:
        print("ERROR: Run llama and qwen steps first!")
        return

    print(f"STEP 3: Consensus (union of Llama + Qwen)")
    print(f"  Llama YES: {sum(1 for p in llama.values() if p['is_hazard'] == 'yes')}")
    print(f"  Qwen YES: {sum(1 for p in qwen.values() if p['is_hazard'] == 'yes')}")

    with CONSENSUS_PREDS.open("w", encoding="utf-8") as f:
        for rec in records:
            pid = rec["id"]
            l = llama.get(pid, {}).get("is_hazard", "no")
            q = qwen.get(pid, {}).get("is_hazard", "no")
            consensus = "yes" if (l == "yes" or q == "yes") else "no"
            source = []
            if l == "yes": source.append("llama")
            if q == "yes": source.append("qwen")

            out = {"id": pid, "is_hazard": consensus, "source": ",".join(source) if source else "none"}
            f.write(json.dumps(out) + "\n")

    consensus = load_predictions(CONSENSUS_PREDS)
    total_yes = sum(1 for p in consensus.values() if p["is_hazard"] == "yes")
    print(f"  Consensus YES: {total_yes}")


# ═══════════════════════════════════════════════════════════════
# STEP 4: Context engineering on consensus-NO
# ═══════════════════════════════════════════════════════════════
def run_context():
    records = load_paragraphs()
    consensus = load_predictions(CONSENSUS_PREDS)

    if not consensus:
        print("ERROR: Run consensus step first!")
        return

    done_ids = load_done_ids(CONTEXT_PREDS)

    # Find consensus-NO with risk score >= 3
    candidates = []
    for rec in records:
        pid = rec["id"]
        if pid in done_ids:
            continue
        if consensus.get(pid, {}).get("is_hazard", "no") == "no":
            score = compute_risk_score(rec["paragraph"], rec.get("section_title", ""))
            if score >= 3:
                candidates.append((rec, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"STEP 4: Context engineering — {len(candidates)} candidates (score >= 3)")

    # Build index for surrounding context
    rec_by_id = {r["id"]: (i, r) for i, r in enumerate(records)}

    with CONTEXT_PREDS.open("a", encoding="utf-8") as f:
        for rec, score in candidates:
            pid = rec["id"]
            idx = rec_by_id[pid][0]

            # Get surrounding paragraphs
            before = [records[i] for i in range(max(0, idx - 2), idx)]
            after = [records[i] for i in range(idx + 1, min(len(records), idx + 3))]

            context_parts = []
            if before:
                context_parts.append("PRECEDING:")
                for bp in before:
                    context_parts.append(f"  [{bp['section_id']}] {bp['paragraph'][:300]}")
            if after:
                context_parts.append("FOLLOWING:")
                for ap in after:
                    context_parts.append(f"  [{ap['section_id']}] {ap['paragraph'][:300]}")

            section_note = ""
            st = rec.get("section_title", "").lower()
            if any(h in st for h in HIGH_RISK_SECTIONS):
                section_note = f"\nNOTE: Section \"{rec['section_title']}\" typically contains security-relevant procedures.\n"

            system = "You are a strict LTE protocol security analyst. Return ONLY JSON: {\"is_hazard\":\"yes\"} or {\"is_hazard\":\"no\"}."
            user = f"""
Does the TARGET PARAGRAPH contain a HAZARD INDICATOR?

Consider surrounding context. If the target INITIATES an action whose destructive
consequences appear in neighboring paragraphs, it IS a hazard.

RETURN "yes" if: credential deletion, bearer deactivation, procedure abort,
context reset, state transition to DEREGISTERED, message discard, or
procedure initiation leading to destructive consequences in context.

RETURN "no" if: message format, state definition, normal procedure, general description.

SECTION: {rec['section_id']} — {rec.get('section_title', '')}
{section_note}
{chr(10).join(context_parts)}

TARGET PARAGRAPH:
\"\"\"{rec['paragraph']}\"\"\"

Return ONLY: {{"is_hazard":"yes"}} or {{"is_hazard":"no"}}
"""
            result = call_qwen(system, user)
            is_hazard = str(result.get("is_hazard", "no")).lower()
            if is_hazard not in ["yes", "no"]:
                is_hazard = "no"

            out = {"id": pid, "is_hazard": is_hazard, "risk_score": score}
            f.write(json.dumps(out) + "\n")
            f.flush()

            status = f"YES (score={score})" if is_hazard == "yes" else f"no (score={score})"
            print(f"  ID {pid} [{rec.get('section_title', '')[:40]}]: {status}")
            time.sleep(0.1)

    context = load_predictions(CONTEXT_PREDS)
    rescued = sum(1 for p in context.values() if p["is_hazard"] == "yes")
    print(f"\nContext engineering rescued: {rescued} paragraphs")


# ═══════════════════════════════════════════════════════════════
# STEP 5: Validate context-rescued paragraphs
# ═══════════════════════════════════════════════════════════════
def run_validate():
    records = load_paragraphs()
    consensus = load_predictions(CONSENSUS_PREDS)
    context = load_predictions(CONTEXT_PREDS)

    if not consensus or not context:
        print("ERROR: Run consensus and context steps first!")
        return

    # Find context-rescued paragraphs (consensus=NO, context=YES)
    rescued = []
    for rec in records:
        pid = rec["id"]
        c = consensus.get(pid, {}).get("is_hazard", "no")
        x = context.get(pid, {}).get("is_hazard", "no")
        if c == "no" and x == "yes":
            rescued.append(rec)

    print(f"STEP 5: Validating {len(rescued)} context-rescued paragraphs")

    done_ids = load_done_ids(VALIDATED_PREDS)

    with VALIDATED_PREDS.open("a", encoding="utf-8") as f:
        for rec in rescued:
            pid = rec["id"]
            if pid in done_ids:
                continue

            system = "You are a skeptical security reviewer. Return ONLY JSON."
            user = f"""
Another analyst flagged this as a hazard indicator. Verify if it's REAL or FALSE POSITIVE.

Section: {rec.get('section_title', '')}
Paragraph: \"\"\"{rec['paragraph']}\"\"\"

FALSE POSITIVE if: just a description/definition, only SENDING a message,
general purpose statement, normal successful procedure, editor's note.

REAL HAZARD if: specific trigger causes credential deletion, bearer deactivation,
procedure abort, context reset, or message discard.

Return: {{"is_valid":"yes","reason":"..."}} or {{"is_valid":"no","reason":"..."}}
"""
            result = call_qwen(system, user)
            is_valid = str(result.get("is_valid", "no")).lower()
            reason = str(result.get("reason", ""))[:80]

            out = {"id": pid, "is_valid": is_valid, "reason": reason}
            f.write(json.dumps(out) + "\n")
            f.flush()

            status = f"CONFIRMED: {reason}" if is_valid == "yes" else f"REJECTED: {reason}"
            print(f"  ID {pid}: {status}")
            time.sleep(0.1)

    validated = load_predictions(VALIDATED_PREDS)
    confirmed = sum(1 for p in validated.values() if p.get("is_valid") == "yes")
    rejected = sum(1 for p in validated.values() if p.get("is_valid") != "yes")
    print(f"\nValidation: {confirmed} confirmed, {rejected} rejected")


# ═══════════════════════════════════════════════════════════════
# STEP 6: Generate pre-annotated CSV for manual review
# ═══════════════════════════════════════════════════════════════
def run_report():
    records = load_paragraphs()
    llama = load_predictions(LLAMA_PREDS)
    qwen = load_predictions(QWEN_PREDS)
    consensus = load_predictions(CONSENSUS_PREDS)
    context = load_predictions(CONTEXT_PREDS)
    validated = load_predictions(VALIDATED_PREDS)

    print(f"STEP 6: Generating pre-annotated CSV")

    with ANNOTATED_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "section_id", "section_title", "paragraph",
            "llama", "qwen", "consensus", "context_rescued", "validated",
            "pipeline_prediction", "confidence",
            "is_hazard", "notes"
        ])
        writer.writeheader()

        stats = {"high_conf_yes": 0, "medium_conf_yes": 0, "low_conf_no": 0, "high_conf_no": 0}

        for rec in records:
            pid = rec["id"]
            l = llama.get(pid, {}).get("is_hazard", "?")
            q = qwen.get(pid, {}).get("is_hazard", "?")
            c = consensus.get(pid, {}).get("is_hazard", "?")
            ctx = context.get(pid, {}).get("is_hazard", "")
            val = validated.get(pid, {}).get("is_valid", "")

            # Determine pipeline prediction and confidence
            if c == "yes":
                prediction = "yes"
                confidence = "HIGH"
                stats["high_conf_yes"] += 1
            elif ctx == "yes" and val == "yes":
                prediction = "yes"
                confidence = "MEDIUM"
                stats["medium_conf_yes"] += 1
            elif ctx == "yes" and val != "yes":
                prediction = "no"
                confidence = "LOW"
                stats["low_conf_no"] += 1
            else:
                prediction = "no"
                confidence = "HIGH"
                stats["high_conf_no"] += 1

            writer.writerow({
                "id": pid,
                "section_id": rec["section_id"],
                "section_title": rec["section_title"],
                "paragraph": rec["paragraph"],
                "llama": l, "qwen": q, "consensus": c,
                "context_rescued": ctx if ctx else "",
                "validated": val if val else "",
                "pipeline_prediction": prediction,
                "confidence": confidence,
                "is_hazard": "",  # For manual annotation
                "notes": "",
            })

    print(f"\nPipeline summary:")
    print(f"  HIGH confidence YES (consensus): {stats['high_conf_yes']}")
    print(f"  MEDIUM confidence YES (context+validated): {stats['medium_conf_yes']}")
    print(f"  LOW confidence NO (context=yes but validation=no): {stats['low_conf_no']}")
    print(f"  HIGH confidence NO: {stats['high_conf_no']}")
    print(f"\nFor manual review:")
    print(f"  Must review: {stats['high_conf_yes'] + stats['medium_conf_yes']} predicted YES")
    print(f"  Should spot-check: ~50 random from {stats['high_conf_no']} predicted NO")
    print(f"  Total manual effort: ~{stats['high_conf_yes'] + stats['medium_conf_yes'] + 50} paragraphs")
    print(f"\nSaved to {ANNOTATED_CSV}")
    print(f"Open in Excel, fill the 'is_hazard' column, focus on predicted YES first.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/pipeline_v9.py llama      # Step 1: Llama (needs Ollama)")
        print("  python src/pipeline_v9.py qwen       # Step 2: Qwen (needs SSH tunnel)")
        print("  python src/pipeline_v9.py consensus   # Step 3: Consensus")
        print("  python src/pipeline_v9.py context     # Step 4: Context engineering")
        print("  python src/pipeline_v9.py validate    # Step 5: Validate rescues")
        print("  python src/pipeline_v9.py report      # Step 6: Generate annotated CSV")
        print("\nRun steps in order. Each supports resume on crash.")
        return

    step = sys.argv[1].lower()

    if step == "llama":
        run_llama()
    elif step == "qwen":
        run_qwen()
    elif step == "consensus":
        run_consensus()
    elif step == "context":
        run_context()
    elif step == "validate":
        run_validate()
    elif step == "report":
        run_report()
    else:
        print(f"Unknown step: {step}")


if __name__ == "__main__":
    main()