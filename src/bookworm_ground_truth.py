"""
BOOKWORM GROUND TRUTH MATCHING
================================
1. Extracts 192 HIs from HI_all_web3.pdf (87 TP + 105 FP)
2. Fuzzy-matches each TP HI against ingested v16.6.0 paragraphs
3. Labels paragraphs: "yes" if contains a Bookworm TP HI, "no" otherwise
4. Outputs gold standard CSV ready for pipeline evaluation

Usage:
  python src/bookworm_ground_truth.py extract    # Step 1: Extract HIs from PDF
  python src/bookworm_ground_truth.py match       # Step 2: Match against paragraphs
  python src/bookworm_ground_truth.py report      # Step 3: Show matching report
"""
import pdfplumber
import re
import json
import csv
import sys
from pathlib import Path
from difflib import SequenceMatcher

# ── Paths (update these for your setup) ──
HI_ALL_PDF = Path("data/raw_specs/HI_all_web3.pdf")      # Bookworm ground truth
T1_PDF = Path("data/raw_specs/T1_web4.pdf")               # T1: abort procedure HIs
T2_PDF = Path("data/raw_specs/T2_web5.pdf")               # T2: consider USIM HIs
PARAGRAPHS_JSONL = Path("data/processed/TS_24_301_v16_paragraphs.jsonl")  # Ingested v16.6.0

# If the PDFs are in the project root instead:
if not HI_ALL_PDF.exists():
    HI_ALL_PDF = Path("HI_all_web3.pdf")
if not T1_PDF.exists():
    T1_PDF = Path("T1_web4.pdf")
if not T2_PDF.exists():
    T2_PDF = Path("T2_web5.pdf")

# Outputs
EXTRACTED_HIS = Path("data/processed/bookworm_his_extracted.jsonl")
GOLD_CSV = Path("data/processed/bookworm_gold_standard.csv")
MATCH_REPORT = Path("data/processed/bookworm_match_report.txt")


def normalize(text):
    """Normalize text for matching: lowercase, collapse whitespace, remove special chars."""
    t = text.lower()
    t = re.sub(r'[ï,·\u2013\u2014\u2018\u2019\u201c\u201d]', ' ', t)  # unicode chars
    t = re.sub(r'[^\w\s#().,;:\-/]', ' ', t)  # keep basic punctuation
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_key_phrases(text):
    """Extract distinctive phrases for matching (message names, actions, cause codes)."""
    phrases = []
    
    # Message names (ALL CAPS multi-word)
    messages = re.findall(r'[A-Z][A-Z ]{3,}[A-Z]', text)
    phrases.extend([m.strip() for m in messages])
    
    # Cause codes like "#3", "#11", "#42"
    causes = re.findall(r'#\d+', text)
    phrases.extend(causes)
    
    # Key action verbs with objects
    actions = re.findall(r'shall\s+(?:delete|abort|discard|reset|deactivate|release|stop|consider|invalidate|remove)\s+\w+(?:\s+\w+)?', text, re.I)
    phrases.extend(actions)
    
    # State names
    states = re.findall(r'EMM-\w+(?:\.\w+)?|BEARER CONTEXT \w+', text)
    phrases.extend(states)
    
    # Timer references
    timers = re.findall(r'T\d{4}', text)
    phrases.extend(timers)
    
    return phrases


def similarity_score(hi_text, para_text):
    """Calculate similarity between HI text and paragraph using multiple strategies."""
    hi_norm = normalize(hi_text)
    para_norm = normalize(para_text)
    
    # Strategy 1: Exact substring match
    if hi_norm in para_norm:
        return 1.0, "exact_substring"
    
    # Strategy 2: Check if all key phrases from HI appear in paragraph
    hi_phrases = extract_key_phrases(hi_text)
    if hi_phrases:
        para_lower = para_text.lower()
        matched_phrases = sum(1 for p in hi_phrases if p.lower() in para_lower)
        phrase_ratio = matched_phrases / len(hi_phrases)
        if phrase_ratio >= 0.8:
            return 0.9, f"key_phrases({matched_phrases}/{len(hi_phrases)})"
    
    # Strategy 3: Token overlap (Jaccard-like)
    hi_tokens = set(hi_norm.split())
    para_tokens = set(para_norm.split())
    if hi_tokens:
        # Remove common stop words
        stops = {'the', 'a', 'an', 'and', 'or', 'if', 'in', 'to', 'of', 'for', 'is',
                'shall', 'be', 'by', 'on', 'at', 'as', 'it', 'that', 'this', 'with',
                'not', 'has', 'was', 'are', 'from'}
        hi_content = hi_tokens - stops
        para_content = para_tokens - stops
        if hi_content:
            overlap = len(hi_content & para_content)
            ratio = overlap / len(hi_content)
            if ratio >= 0.7:
                return ratio * 0.85, f"token_overlap({overlap}/{len(hi_content)})"
    
    # Strategy 4: Sequence matcher for partial matches
    ratio = SequenceMatcher(None, hi_norm[:200], para_norm[:500]).ratio()
    if ratio >= 0.6:
        return ratio * 0.8, f"sequence_match({ratio:.2f})"
    
    return 0.0, "no_match"


def extract_his_from_pdf(pdf_path):
    """Extract HI entries from HI_all_web3.pdf."""
    his = []
    
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"
    
    # The HI_all document has numbered entries
    # Each HI starts with a number at the beginning of a section
    # Parse by looking for HI number patterns
    
    lines = full_text.split("\n")
    current_hi = None
    current_text = []
    hi_number = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip page markers
        if line.startswith("FP1") or line.startswith("FP2"):
            continue
        
        # Check if this is a new HI entry (starts with a number)
        # HI entries in the document start with the HI number
        m = re.match(r'^(\d{1,3})\s+(.+)', line)
        if m:
            num = int(m.group(1))
            rest = m.group(2)
            
            # Verify it's a real HI number (sequential, reasonable range)
            if 1 <= num <= 200 and (num == hi_number + 1 or num == 1 or hi_number == 0):
                # Save previous HI
                if current_hi is not None:
                    his.append({
                        "hi_no": current_hi,
                        "text": " ".join(current_text).strip(),
                    })
                
                current_hi = num
                hi_number = num
                current_text = [rest]
                continue
        
        # Continuation of current HI
        if current_hi is not None:
            current_text.append(line)
    
    # Save last HI
    if current_hi is not None:
        his.append({
            "hi_no": current_hi,
            "text": " ".join(current_text).strip(),
        })
    
    return his


def extract_tp_fp_labels(his, t1_pdf=None, t2_pdf=None):
    """
    Determine TP/FP labels. 
    From the document structure:
    - HIs in HI_all that appear in T1 or T2 with checkmarks are validated TPs
    - The document has markers like ✗1, ✗2, ✗3, ✗4, ✗5 for FP reasons
    - HIs without ✗ markers are TPs
    """
    for hi in his:
        text = hi["text"]
        # Check for FP markers in the text
        has_fp_marker = bool(re.search(r'[✗✘×]|\\u2717|\\u2718', text))
        
        # Also check for FP reason codes at end of text
        fp_patterns = [
            r'✗\d', r'FP\d', 
            r'This is due to that the message',
            r'This is due to the ambiguity',
            r'This is due to that the recipient',
            r'This is due to that the risky operation',
            r'This is due to the simulator',
        ]
        for pat in fp_patterns:
            if re.search(pat, text):
                has_fp_marker = True
                break
        
        hi["is_tp"] = not has_fp_marker
    
    return his


def run_extract():
    """Step 1: Extract HIs from Bookworm PDFs."""
    print("=" * 60)
    print("STEP 1: Extracting Bookworm HIs from PDFs")
    print("=" * 60)
    
    if not HI_ALL_PDF.exists():
        print(f"ERROR: Cannot find {HI_ALL_PDF}")
        print(f"Copy HI_all_web3.pdf to data/raw_specs/ or project root")
        return
    
    # Extract from HI_all
    print(f"Reading {HI_ALL_PDF}...")
    his = extract_his_from_pdf(HI_ALL_PDF)
    print(f"Extracted {len(his)} HI entries")
    
    # Label TP/FP
    his = extract_tp_fp_labels(his)
    tp_count = sum(1 for h in his if h["is_tp"])
    fp_count = sum(1 for h in his if not h["is_tp"])
    print(f"Labeled: {tp_count} TP, {fp_count} FP")
    
    if tp_count != 87:
        print(f"WARNING: Expected 87 TPs but found {tp_count}")
        print(f"The TP/FP detection may need adjustment.")
        print(f"Showing first 5 TPs and first 5 FPs for verification:")
        print(f"\nFirst 5 TPs:")
        for h in [x for x in his if x["is_tp"]][:5]:
            print(f"  HI {h['hi_no']}: {h['text'][:100]}...")
        print(f"\nFirst 5 FPs:")
        for h in [x for x in his if not x["is_tp"]][:5]:
            print(f"  HI {h['hi_no']}: {h['text'][:100]}...")
    
    # Save extracted HIs
    EXTRACTED_HIS.parent.mkdir(parents=True, exist_ok=True)
    with EXTRACTED_HIS.open("w", encoding="utf-8") as f:
        for h in his:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    
    print(f"\nSaved to {EXTRACTED_HIS}")
    print(f"NEXT: Verify the TP/FP labels are correct, then run 'match'")


def run_match():
    """Step 2: Match Bookworm TPs against ingested paragraphs."""
    print("=" * 60)
    print("STEP 2: Matching Bookworm TPs to Paragraphs")
    print("=" * 60)
    
    # Load extracted HIs
    if not EXTRACTED_HIS.exists():
        print("ERROR: Run 'extract' step first!")
        return
    
    his = []
    with EXTRACTED_HIS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                his.append(json.loads(line))
    
    tp_his = [h for h in his if h["is_tp"]]
    print(f"Loaded {len(tp_his)} TP HIs")
    
    # Load paragraphs
    if not PARAGRAPHS_JSONL.exists():
        print(f"ERROR: Cannot find {PARAGRAPHS_JSONL}")
        print(f"Run ingestion on TS 24.301 v16.6.0 first!")
        return
    
    paragraphs = []
    with PARAGRAPHS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                paragraphs.append(json.loads(line))
    
    print(f"Loaded {len(paragraphs)} paragraphs")
    
    # Match each TP HI to paragraphs
    print(f"\nMatching {len(tp_his)} TPs against {len(paragraphs)} paragraphs...")
    
    matched_paras = set()  # paragraph IDs that contain a TP HI
    match_details = []     # detailed match info
    unmatched_his = []     # HIs that couldn't be matched
    
    for hi in tp_his:
        hi_text = hi["text"]
        best_score = 0
        best_para_id = None
        best_method = ""
        
        for para in paragraphs:
            score, method = similarity_score(hi_text, para["paragraph"])
            if score > best_score:
                best_score = score
                best_para_id = para["id"]
                best_method = method
        
        if best_score >= 0.5:
            matched_paras.add(best_para_id)
            match_details.append({
                "hi_no": hi["hi_no"],
                "para_id": best_para_id,
                "score": round(best_score, 3),
                "method": best_method,
                "hi_text": hi_text[:100],
                "para_text": paragraphs[best_para_id]["paragraph"][:100] if best_para_id < len(paragraphs) else "",
            })
            print(f"  HI {hi['hi_no']:>3d} → Para {best_para_id:>4d} (score={best_score:.3f}, {best_method})")
        else:
            unmatched_his.append(hi)
            print(f"  HI {hi['hi_no']:>3d} → NO MATCH (best={best_score:.3f})")
    
    matched_count = len(match_details)
    print(f"\nMatched: {matched_count}/{len(tp_his)} TPs ({matched_count/len(tp_his)*100:.1f}%)")
    print(f"Unmatched: {len(unmatched_his)}")
    print(f"Unique paragraphs matched: {len(matched_paras)}")
    
    # Show unmatched HIs for debugging
    if unmatched_his:
        print(f"\n--- Unmatched HIs (need manual review) ---")
        for hi in unmatched_his:
            print(f"  HI {hi['hi_no']}: {hi['text'][:120]}...")
    
    # Create gold standard CSV
    print(f"\nCreating gold standard CSV...")
    
    with GOLD_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "section_id", "section_title", "paragraph",
            "is_hazard", "matched_his", "notes"
        ])
        writer.writeheader()
        
        for para in paragraphs:
            pid = para["id"]
            is_hazard = "yes" if pid in matched_paras else "no"
            
            # Find which HIs matched this paragraph
            matching_his = [m["hi_no"] for m in match_details if m["para_id"] == pid]
            
            writer.writerow({
                "id": pid,
                "section_id": para.get("section_id", ""),
                "section_title": para.get("section_title", ""),
                "paragraph": para["paragraph"],
                "is_hazard": is_hazard,
                "matched_his": ",".join(str(h) for h in matching_his),
                "notes": "",
            })
    
    total_yes = len(matched_paras)
    total_no = len(paragraphs) - total_yes
    print(f"\nGold standard created:")
    print(f"  YES (hazard): {total_yes} paragraphs")
    print(f"  NO (not hazard): {total_no} paragraphs")
    print(f"  Positive rate: {total_yes/len(paragraphs)*100:.1f}%")
    print(f"\nSaved to {GOLD_CSV}")
    
    # Save match details
    match_report_path = Path("data/processed/bookworm_match_details.jsonl")
    with match_report_path.open("w", encoding="utf-8") as f:
        for m in match_details:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    
    print(f"Match details saved to {match_report_path}")


def run_report():
    """Step 3: Show matching report and quality assessment."""
    print("=" * 60)
    print("STEP 3: Matching Quality Report")
    print("=" * 60)
    
    if not GOLD_CSV.exists():
        print("ERROR: Run 'match' step first!")
        return
    
    # Load gold standard
    with GOLD_CSV.open("r", encoding="utf-8") as f:
        gold = list(csv.DictReader(f))
    
    yes_count = sum(1 for r in gold if r["is_hazard"] == "yes")
    no_count = sum(1 for r in gold if r["is_hazard"] == "no")
    
    print(f"Total paragraphs: {len(gold)}")
    print(f"Labeled YES: {yes_count}")
    print(f"Labeled NO: {no_count}")
    print(f"Positive rate: {yes_count/len(gold)*100:.1f}%")
    
    # Load match details
    match_path = Path("data/processed/bookworm_match_details.jsonl")
    if match_path.exists():
        matches = []
        with match_path.open() as f:
            for line in f:
                if line.strip():
                    matches.append(json.loads(line))
        
        # Quality breakdown
        exact = sum(1 for m in matches if m["method"] == "exact_substring")
        phrase = sum(1 for m in matches if "key_phrases" in m["method"])
        token = sum(1 for m in matches if "token_overlap" in m["method"])
        sequence = sum(1 for m in matches if "sequence_match" in m["method"])
        
        print(f"\nMatch quality breakdown:")
        print(f"  Exact substring: {exact}")
        print(f"  Key phrase match: {phrase}")
        print(f"  Token overlap: {token}")
        print(f"  Sequence match: {sequence}")
        
        # Score distribution
        scores = [m["score"] for m in matches]
        if scores:
            print(f"\nMatch score distribution:")
            print(f"  Score >= 0.9: {sum(1 for s in scores if s >= 0.9)}")
            print(f"  Score 0.7-0.9: {sum(1 for s in scores if 0.7 <= s < 0.9)}")
            print(f"  Score 0.5-0.7: {sum(1 for s in scores if 0.5 <= s < 0.7)}")
        
        # Section distribution of matched paragraphs
        print(f"\nHazard paragraphs by section:")
        from collections import Counter
        sections = Counter()
        for r in gold:
            if r["is_hazard"] == "yes":
                sec = r["section_id"].split('.')[0] if r["section_id"] else "?"
                sections[sec] += 1
        for s in sorted(sections.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            print(f"  Section {s}: {sections[s]} HIs")
    
    # Load extracted HIs to check TP count
    if EXTRACTED_HIS.exists():
        his = []
        with EXTRACTED_HIS.open() as f:
            for line in f:
                if line.strip():
                    his.append(json.loads(line))
        tp_count = sum(1 for h in his if h["is_tp"])
        
        print(f"\nBookworm ground truth: {tp_count} TPs")
        print(f"Matched to paragraphs: {yes_count}")
        print(f"Coverage: {yes_count}/{tp_count} = {yes_count/tp_count*100:.1f}%")
        
        if yes_count < tp_count:
            unmatched = tp_count - yes_count
            print(f"\n⚠ {unmatched} TPs not matched to any paragraph!")
            print(f"These may need manual matching or the ingestion may have")
            print(f"missed the relevant paragraphs from the spec.")
    
    print(f"\n{'='*60}")
    print(f"Gold standard ready at: {GOLD_CSV}")
    print(f"Next: Run your pipeline on the paragraphs and evaluate")
    print(f"against this gold standard.")
    print(f"  python src/pipeline_v9.py llama")
    print(f"  python src/pipeline_v9.py qwen")
    print(f"  ... etc")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/bookworm_ground_truth.py extract  # Extract HIs from PDFs")
        print("  python src/bookworm_ground_truth.py match    # Match against paragraphs")
        print("  python src/bookworm_ground_truth.py report   # Show matching report")
        print("\nRun steps in order.")
        print(f"\nExpected files:")
        print(f"  {HI_ALL_PDF}")
        print(f"  {PARAGRAPHS_JSONL}")
        return
    
    step = sys.argv[1].lower()
    
    if step == "extract":
        run_extract()
    elif step == "match":
        run_match()
    elif step == "report":
        run_report()
    else:
        print(f"Unknown step: {step}")


if __name__ == "__main__":
    main()