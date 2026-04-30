"""
IMPROVED INGESTION for TS 24.301 V9.0.0
========================================
Fixes: page headers/footers, cross-page paragraph splits,
single-sentence fragments, cause code merging, deduplication.
"""
import pdfplumber
import re
import json
import csv
from pathlib import Path
from collections import Counter

PDF_PATH = Path("data/raw_specs/TS_24.301.900.pdf")
OUT_JSONL = Path("data/processed/TS_24_301_v9_paragraphs.jsonl")
OUT_CSV = Path("data/processed/annotation_v9_sample.csv")

# Bookworm-scope outputs (sections 4.4, 5.4, 5.5, 6.3-6.6)
BOOKWORM_JSONL = Path("data/processed/TS_24_301_v9_bookworm.jsonl")
BOOKWORM_CSV = Path("data/processed/annotation_v9_bookworm.csv")
BOOKWORM_PREFIXES = ['4.4', '5.4', '5.5', '6.3', '6.4', '6.5', '6.6']

SECTION_RE = re.compile(r'^(\d+(?:\.\d+)*)\s{1,4}([A-Z][\w\s\(\)\-,;/]+)')
HEADER_RE = re.compile(r'^Release\s+\d+\s+\d+\s+3GPP\s+TS\s+24', re.I)
CAUSE_RE = re.compile(r'^#\d+\s*[\(:]')


def extract_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
    return pages


def strip_header_footer(lines):
    """Remove page headers and footers."""
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if HEADER_RE.match(s):
            continue
        if s == "3GPP" or s == "3GPP TS 24.301":
            continue
        # Standalone page numbers
        if re.match(r'^\d{1,3}$', s):
            continue
        cleaned.append(s)
    return cleaned


def build_raw_stream(pages):
    """Combine all pages into a single stream, joining cross-page text."""
    stream = []
    prev_page_last_line = ""

    for page_num, text in pages:
        lines = text.split("\n")
        lines = strip_header_footer(lines)

        if not lines:
            continue

        # Check if first line of this page continues from previous page
        first_real = ""
        for l in lines:
            if l.strip():
                first_real = l.strip()
                break

        if prev_page_last_line and first_real:
            # If previous page didn't end with period and this page starts lowercase
            # or with a continuation word, merge them
            if (prev_page_last_line[-1] not in '.;:' or
                (first_real and first_real[0].islower()) or
                first_real.startswith(('the ', 'a ', 'an ', 'and ', 'or ', 'if ',
                    'when ', 'upon ', 'shall ', 'may ', 'with ', 'without ',
                    'for ', 'from ', 'to ', 'in ', 'on ', 'by ', 'at ', 'as ',
                    'of ', 'that ', 'which ', 'where ', 'after ', 'before ',
                    'state ', 'timer ', 'procedure ', 'message ', 'counter ',
                    'number ', 'status ', 'attach ', 'detach ', 'service '))):
                # Remove the last blank line if any, and merge
                while stream and stream[-1] == "":
                    stream.pop()
                # Don't add blank line between pages for continuation

        for line in lines:
            stream.append((page_num, line))

        # Track last non-empty line
        for l in reversed(lines):
            if l.strip():
                prev_page_last_line = l.strip()
                break

    return stream


def parse_paragraphs(stream):
    """Parse the line stream into structured paragraphs."""
    records = []
    current_section_id = ""
    current_section_title = ""
    current_lines = []
    current_page = 0
    started = False

    def flush():
        nonlocal current_lines
        if current_lines:
            text = " ".join(current_lines).strip()
            # Clean up extra spaces
            text = re.sub(r'\s+', ' ', text)
            if len(text) >= 40:
                records.append({
                    "section_id": current_section_id,
                    "section_title": current_section_title,
                    "paragraph": text,
                    "page_start": current_page,
                })
            current_lines = []

    for page_num, line in stream:
        line = line.strip() if isinstance(line, str) else ""

        # Skip TOC lines (dotted leaders)
        if '...' in line and re.search(r'\d+\s*$', line):
            continue

        # Skip until Section 1 (actual content, not TOC)
        if not started:
            if line == '1 Scope' or re.match(r'^1\s+Scope\s*$', line):
                started = True
                current_section_id = "1"
                current_section_title = "Scope"
                current_page = page_num
            continue

        # Stop at Annex
        if re.match(r'^Annex\s+[A-Z]', line):
            flush()
            break

        # Blank line = potential paragraph break
        if not line:
            # Only flush if we have substantial content
            if current_lines and len(" ".join(current_lines)) >= 40:
                flush()
            continue

        # Section header
        m = SECTION_RE.match(line)
        if m:
            sec_id = m.group(1)
            sec_title = m.group(2).strip()
            # Verify: real section headers have depth <= 7 and reasonable titles
            parts = sec_id.split('.')
            if len(parts) <= 7 and len(sec_title) > 3 and not sec_title[0].isdigit():
                flush()
                current_section_id = sec_id
                current_section_title = sec_title
                current_page = page_num
                continue

        # Skip figure/table captions
        if re.match(r'^(Figure|Table)\s+\d', line):
            continue

        # Skip TOC lines (dotted leaders)
        if '...' in line and re.search(r'\d+\s*$', line):
            continue

        # Cause code line: merge with following paragraph
        if CAUSE_RE.match(line):
            flush()
            current_lines = [line]
            current_page = page_num
            continue

        # NOTE lines: keep as part of current paragraph
        if re.match(r'^NOTE\s*\d*\s*:', line):
            current_lines.append(line)
            continue

        # Bullet/dash items: keep in current paragraph
        if line.startswith('- ') or re.match(r'^[a-z]\)\s', line):
            current_lines.append(line)
            continue

        # Regular text line
        if current_lines:
            prev = current_lines[-1]
            # Merge if: previous doesn't end with sentence-ender, or this starts lowercase
            if (prev and prev[-1] not in '.;:' and prev[-1] != ')') or \
               (line and line[0].islower()):
                current_lines.append(line)
            else:
                # Check if current paragraph is too short to stand alone
                current_text = " ".join(current_lines)
                if len(current_text) < 80:
                    current_lines.append(line)
                else:
                    flush()
                    current_lines = [line]
                    current_page = page_num
        else:
            current_lines = [line]
            current_page = page_num

    flush()
    return records


def deduplicate(records):
    """Remove exact and near-duplicate paragraphs."""
    seen = set()
    deduped = []
    for r in records:
        norm = re.sub(r'\s+', ' ', r["paragraph"].lower().strip())
        # Use first 100 chars as key to catch near-dupes
        key = norm[:100]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def is_table_content(text):
    """Detect message format/IE tables."""
    indicators = ['IEI', 'Information Element', 'Presence', 'Format',
                  'Length', 'Type/Reference', 'octet', 'Message type']
    count = sum(1 for ind in indicators if ind in text)
    return count >= 3


def quality_filter(records):
    """Remove low-quality paragraphs."""
    filtered = []
    for r in records:
        text = r["paragraph"]
        # Too short
        if len(text) < 40:
            continue
        # Just a reference
        if re.match(r'^See\s+(subclause|clause|section|table|figure|annex)', text, re.I):
            continue
        # Just "Void" or placeholder
        if text.strip().lower() in ['void', 'void.', 'reserved', 'spare']:
            continue
        filtered.append(r)
    return filtered


def main():
    print(f"Reading {PDF_PATH}...")
    pages = extract_pages(PDF_PATH)
    print(f"Pages: {len(pages)}")

    print("Building text stream...")
    stream = build_raw_stream(pages)
    print(f"Lines in stream: {len(stream)}")

    print("Parsing paragraphs...")
    records = parse_paragraphs(stream)
    print(f"Raw paragraphs: {len(records)}")

    print("Deduplicating...")
    records = deduplicate(records)
    print(f"After dedup: {len(records)}")

    print("Quality filtering...")
    records = quality_filter(records)
    print(f"After quality filter: {len(records)}")

    # Mark tables
    for r in records:
        r["is_table"] = is_table_content(r["paragraph"])

    tables = sum(1 for r in records if r["is_table"])
    content = len(records) - tables
    print(f"\nContent paragraphs: {content}")
    print(f"Table paragraphs: {tables}")

    # Section distribution
    sections = Counter(r["section_id"].split('.')[0] for r in records)
    print(f"\nBy top-level section:")
    for s in sorted(sections.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        c = sections[s]
        tc = sum(1 for r in records if r["section_id"].split('.')[0] == s and r["is_table"])
        print(f"  Section {s}: {c} total ({c-tc} content, {tc} tables)")

    # Show HI-rich paragraphs as quality check
    hi_words = ['shall delete', 'shall abort', 'shall discard', 'shall reset',
                'shall deactivate', 'shall invalidate', 'shall release',
                'enter state emm-deregistered', 'enter the state emm-deregistered']
    print(f"\n=== Paragraphs with HI keywords (quality check) ===")
    hi_count = 0
    for r in records:
        if r["is_table"]:
            continue
        lower = r["paragraph"].lower()
        matches = [w for w in hi_words if w in lower]
        if matches:
            hi_count += 1
            if hi_count <= 8:
                print(f"\n  [{r['section_id']} - {r['section_title'][:45]}] (page {r['page_start']})")
                print(f"  Keywords: {', '.join(matches)}")
                print(f"  Text: {r['paragraph'][:180]}...")
    print(f"\nTotal paragraphs with HI keywords: {hi_count}")

    # Paragraph length distribution
    lengths = [len(r["paragraph"]) for r in records if not r["is_table"]]
    if lengths:
        print(f"\nParagraph length stats (content only):")
        print(f"  Min: {min(lengths)} chars")
        print(f"  Max: {max(lengths)} chars")
        print(f"  Avg: {sum(lengths)//len(lengths)} chars")
        print(f"  < 50 chars: {sum(1 for l in lengths if l < 50)}")
        print(f"  50-100 chars: {sum(1 for l in lengths if 50 <= l < 100)}")
        print(f"  100-300 chars: {sum(1 for l in lengths if 100 <= l < 300)}")
        print(f"  300+ chars: {sum(1 for l in lengths if l >= 300)}")
    else:
        print("\nNo content paragraphs found!")

    # Save JSONL
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            rec["id"] = i
            rec["spec"] = "TS 24.301 V9.0.0"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(records)} paragraphs to {OUT_JSONL}")

    # Save annotation CSV (content paragraphs only)
    content_records = [r for r in records if not r["is_table"]]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "section_id", "section_title", "paragraph",
            "is_hazard", "condition", "operation", "state", "hazard_type", "notes"
        ])
        writer.writeheader()
        for i, rec in enumerate(content_records):
            writer.writerow({
                "id": i,
                "section_id": rec["section_id"],
                "section_title": rec["section_title"],
                "paragraph": rec["paragraph"],
                "is_hazard": "", "condition": "", "operation": "",
                "state": "", "hazard_type": "", "notes": "",
            })
    print(f"Saved annotation template: {len(content_records)} content paragraphs to {OUT_CSV}")

    # ── Bookworm-scope subset (sections 4.4, 5.4, 5.5, 6.3-6.6) ──
    print(f"\n{'='*60}")
    print("BOOKWORM-SCOPE SUBSET")
    print(f"{'='*60}")

    bookworm = [r for r in records
                if any(r["section_id"].startswith(p) for p in BOOKWORM_PREFIXES)
                and not r.get("is_table", False)]

    # Re-number IDs
    for i, r in enumerate(bookworm):
        r["id"] = i

    # Save Bookworm JSONL
    with BOOKWORM_JSONL.open("w", encoding="utf-8") as f:
        for r in bookworm:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save Bookworm annotation CSV
    with BOOKWORM_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "section_id", "section_title", "paragraph",
            "is_hazard", "condition", "operation", "state", "hazard_type", "notes"
        ])
        writer.writeheader()
        for r in bookworm:
            writer.writerow({
                "id": r["id"],
                "section_id": r["section_id"],
                "section_title": r["section_title"],
                "paragraph": r["paragraph"],
                "is_hazard": "", "condition": "", "operation": "",
                "state": "", "hazard_type": "", "notes": "",
            })

    # Stats
    bw_hi = sum(1 for r in bookworm if any(w in r["paragraph"].lower() for w in hi_words))
    print(f"Sections: {', '.join(BOOKWORM_PREFIXES)}")
    print(f"Content paragraphs: {len(bookworm)}")
    print(f"Estimated HIs (keyword match): ~{bw_hi}")
    print(f"\nBy section:")
    for prefix in BOOKWORM_PREFIXES:
        paras = [r for r in bookworm if r["section_id"].startswith(prefix)]
        hi_c = sum(1 for r in paras if any(w in r["paragraph"].lower() for w in hi_words))
        print(f"  {prefix}: {len(paras)} paragraphs, ~{hi_c} HIs")
    print(f"\nSaved to:")
    print(f"  {BOOKWORM_JSONL}")
    print(f"  {BOOKWORM_CSV}")


if __name__ == "__main__":
    main()