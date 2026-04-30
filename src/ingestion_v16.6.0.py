"""
INGESTION FOR TS 24.301 v16.6.0 (Release 16, Version 6.0)
==========================================================
Adapted from ingestion.py (the cleaner, proven approach).
Focuses on: section detection, paragraph aggregation, deduplication.
Avoids: over-aggressive cross-page merging, cause-code special handling, artifact generation.

Tested logic from ingestion.py:
  ✓ Clean section header regex matching
  ✓ TOC/header/footer noise removal
  ✓ Simple paragraph break on blank lines
  ✓ Minimum paragraph length threshold (20 chars)
  ✓ Table content detection (IEI, Format, etc.)
  ✓ Deduplication by normalized text
"""

import pdfplumber
import re
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter

# ========== CONFIGURATION ==========
PDF_PATH = Path("data/raw_specs/TS_24.301.16.6.0.pdf")
OUT_JSONL = Path("data/processed/TS_24_301_v16.6.0_paragraphs.jsonl")
OUT_CSV = Path("data/processed/annotation_v16.6.0_sample.csv")

# Bookworm-scope outputs (sections 4.4, 5.4, 5.5, 6.3-6.6)
BOOKWORM_JSONL = Path("data/processed/TS_24_301_v16.6.0_bookworm.jsonl")
BOOKWORM_CSV = Path("data/processed/annotation_v16.6.0_bookworm.csv")
BOOKWORM_PREFIXES = ['4.4', '5.4', '5.5', '6.3', '6.4', '6.5', '6.6']

# ========== PATTERNS ==========
# Matches real section headers like "5.4.4  EPS attach procedure"
# but NOT TOC lines (which have dotted leaders or page numbers at end)
SECTION_PATTERN = re.compile(r"^(\d+(\.\d+)*)\s+(.+)$")

# Page header/footer noise
PAGE_HEADER_PATTERN = re.compile(
    r"^(3GPP\s*$|Release\s+\d+\s+\d+\s+3GPP|3GPP\s+TS\s+24\.301|TS\s+24\.301)",
    re.IGNORECASE
)

@dataclass
class ParagraphRecord:
    spec: str
    section_id: str
    section_title: str
    paragraph: str
    page_start: int = 0
    is_table: bool = False
    id: int = 0  # Will be set during processing

# ========== CORE FUNCTIONS ==========

def pdf_to_text(pdf_path: Path) -> list:
    """Extract pages from PDF with page numbers."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
    return pages


def is_toc_line(line: str) -> bool:
    """Detect TOC entries like '5.1 Overview ....... 13'"""
    if '...' in line:
        return True
    # Lines ending with a page number after lots of spaces
    if re.search(r'\s{3,}\d{1,3}\s*$', line):
        return True
    return False


def is_page_noise(line: str) -> bool:
    """Detect page headers/footers"""
    if PAGE_HEADER_PATTERN.match(line):
        return True
    # Standalone page numbers
    if re.match(r'^\d{1,3}\s*$', line):
        return True
    # Figure/Table captions (often noise)
    if re.match(r'^(Figure|Table|Annex)\s+', line):
        return True
    return False


def is_table_content(paragraph: str) -> bool:
    """Detect message format tables (IEI tables, etc.)"""
    indicators = [
        'IEI' in paragraph and 'Information Element' in paragraph,
        'Presence' in paragraph and 'Format' in paragraph and 'Length' in paragraph,
        paragraph.count('\t') > 3,  # tab-heavy = likely table
    ]
    return any(indicators)


def parse_sections(pages: list, spec_name: str = "TS 24.301 v16.6.0"):
    """Parse pages into structured paragraphs with section tracking."""
    records = []
    current_section_id = ""
    current_section_title = ""
    current_para_lines = []
    current_page = 0
    started = False  # skip everything before Section 1

    for page_num, page_text in pages:
        lines = page_text.splitlines()

        for line in lines:
            line_stripped = line.strip()

            # Skip until we reach "1 Scope" (the real start of the spec)
            if not started:
                if re.match(r'^1\s+Scope', line_stripped) and not is_toc_line(line_stripped):
                    started = True
                    current_section_id = "1"
                    current_section_title = "Scope"
                    current_page = page_num
                continue

            # Skip blank lines (paragraph break)
            if not line_stripped:
                if current_para_lines:
                    paragraph_text = " ".join(current_para_lines).strip()
                    # Normalize whitespace
                    paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                    if len(paragraph_text) >= 40:  # Slightly higher threshold than ingestion.py
                        records.append(
                            ParagraphRecord(
                                spec=spec_name,
                                section_id=current_section_id,
                                section_title=current_section_title,
                                paragraph=paragraph_text,
                                page_start=current_page,
                                is_table=is_table_content(paragraph_text),
                            )
                        )
                    current_para_lines = []
                continue

            # Skip page noise
            if is_page_noise(line_stripped):
                continue

            # Skip TOC lines
            if is_toc_line(line_stripped):
                continue

            # Check for section header
            m = SECTION_PATTERN.match(line_stripped)
            if m:
                # Flush current paragraph
                if current_para_lines:
                    paragraph_text = " ".join(current_para_lines).strip()
                    paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                    if len(paragraph_text) >= 40:
                        records.append(
                            ParagraphRecord(
                                spec=spec_name,
                                section_id=current_section_id,
                                section_title=current_section_title,
                                paragraph=paragraph_text,
                                page_start=current_page,
                                is_table=is_table_content(paragraph_text),
                            )
                        )
                    current_para_lines = []

                candidate_title = m.group(3).strip()
                # Make sure this isn't a TOC line that slipped through
                if not is_toc_line(candidate_title):
                    current_section_id = m.group(1)
                    current_section_title = candidate_title
                    current_page = page_num
            else:
                current_para_lines.append(line_stripped)

    # Flush last paragraph
    if current_para_lines:
        paragraph_text = " ".join(current_para_lines).strip()
        paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
        if len(paragraph_text) >= 40:
            records.append(
                ParagraphRecord(
                    spec=spec_name,
                    section_id=current_section_id,
                    section_title=current_section_title,
                    paragraph=paragraph_text,
                    page_start=current_page,
                    is_table=is_table_content(paragraph_text),
                )
            )

    return records


def deduplicate(records):
    """Remove exact and near-duplicate paragraphs."""
    seen = set()
    deduped = []
    for r in records:
        # Normalize: lowercase, collapse whitespace
        norm = re.sub(r'\s+', ' ', r.paragraph.lower().strip())
        # Use first 100 chars as key to catch near-dupes
        key = norm[:100]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def quality_filter(records):
    """Remove low-quality paragraphs."""
    filtered = []
    for r in records:
        text = r.paragraph
        
        # Too short (already filtered by >= 40, but extra check)
        if len(text) < 40:
            continue
        
        # Just a reference
        if re.match(r'^See\s+(subclause|clause|section|table|figure|annex)', text, re.I):
            continue
        
        # Just "Void" or placeholder
        if text.strip().lower() in ['void', 'void.', 'reserved', 'spare']:
            continue
        
        # Single sentence fragment (likely artifact)
        if text.count('.') == 0 and len(text) < 80:
            continue
        
        filtered.append(r)
    
    return filtered


def main():
    print(f"Reading {PDF_PATH}...")
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")
    
    pages = pdf_to_text(PDF_PATH)
    print(f"  ✓ Extracted {len(pages)} pages")

    print("Parsing paragraphs...")
    records = parse_sections(pages)
    print(f"  ✓ Raw paragraphs: {len(records)}")

    print("Deduplicating...")
    records = deduplicate(records)
    print(f"  ✓ After dedup: {len(records)}")

    print("Quality filtering...")
    records = quality_filter(records)
    print(f"  ✓ After quality filter: {len(records)}")

    # Add IDs
    for i, r in enumerate(records):
        r.id = i

    # Statistics
    tables = sum(1 for r in records if r.is_table)
    content = len(records) - tables
    print(f"\n{'='*60}")
    print("INGESTION STATISTICS")
    print(f"{'='*60}")
    print(f"Content paragraphs: {content}")
    print(f"Table paragraphs: {tables}")

    # Section distribution
    sections = Counter(r.section_id.split('.')[0] for r in records)
    print(f"\nBy top-level section:")
    for s in sorted(sections.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        c = sections[s]
        tc = sum(1 for r in records if r.section_id.split('.')[0] == s and r.is_table)
        print(f"  Section {s}: {c} total ({c-tc} content, {tc} tables)")

    # Show HI-rich paragraphs as quality check
    hi_words = [
        'shall delete', 'shall abort', 'shall discard', 'shall reset',
        'shall deactivate', 'shall invalidate', 'shall release',
        'enter state emm-deregistered', 'enter the state emm-deregistered',
        'release resources', 'reject', 'abort procedure'
    ]
    print(f"\n{'='*60}")
    print("HI KEYWORD DETECTION (Quality Check)")
    print(f"{'='*60}")
    hi_count = 0
    for r in records:
        if r.is_table:
            continue
        lower = r.paragraph.lower()
        matches = [w for w in hi_words if w in lower]
        if matches:
            hi_count += 1
            if hi_count <= 10:
                print(f"\n  [{r.section_id} - {r.section_title[:40]}] (page {r.page_start})")
                print(f"  Keywords: {', '.join(matches)}")
                print(f"  Text: {r.paragraph[:150]}...")
    print(f"\nTotal paragraphs with HI keywords: {hi_count} / {content} content")

    # Paragraph length distribution
    lengths = [len(r.paragraph) for r in records if not r.is_table]
    if lengths:
        print(f"\n{'='*60}")
        print("PARAGRAPH LENGTH DISTRIBUTION (Content Only)")
        print(f"{'='*60}")
        print(f"  Min: {min(lengths)} chars")
        print(f"  Max: {max(lengths)} chars")
        print(f"  Avg: {sum(lengths)//len(lengths)} chars")
        print(f"  < 100 chars: {sum(1 for l in lengths if l < 100)}")
        print(f"  100-300 chars: {sum(1 for l in lengths if 100 <= l < 300)}")
        print(f"  300-500 chars: {sum(1 for l in lengths if 300 <= l < 500)}")
        print(f"  500+ chars: {sum(1 for l in lengths if l >= 500)}")

    # Save JSONL
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"\n✓ Saved {len(records)} paragraphs to {OUT_JSONL}")

    # Save annotation CSV (content paragraphs only)
    content_records = [r for r in records if not r.is_table]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "section_id", "section_title", "paragraph",
            "is_hazard", "condition", "operation", "state", "hazard_type", "notes"
        ])
        writer.writeheader()
        for r in content_records:
            writer.writerow({
                "id": r.id,
                "section_id": r.section_id,
                "section_title": r.section_title,
                "paragraph": r.paragraph,
                "is_hazard": "", "condition": "", "operation": "",
                "state": "", "hazard_type": "", "notes": "",
            })
    print(f"✓ Saved annotation template: {len(content_records)} to {OUT_CSV}")

    # ── BOOKWORM-SCOPE SUBSET ──
    print(f"\n{'='*60}")
    print("BOOKWORM-SCOPE SUBSET (Sections 4.4, 5.4, 5.5, 6.3-6.6)")
    print(f"{'='*60}")

    bookworm = [r for r in records
                if any(r.section_id.startswith(p) for p in BOOKWORM_PREFIXES)
                and not r.is_table]

    # Re-number IDs for Bookworm subset
    for i, r in enumerate(bookworm):
        r.id = i

    # Save Bookworm JSONL
    with BOOKWORM_JSONL.open("w", encoding="utf-8") as f:
        for r in bookworm:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Save Bookworm annotation CSV
    with BOOKWORM_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "section_id", "section_title", "paragraph",
            "is_hazard", "condition", "operation", "state", "hazard_type", "notes"
        ])
        writer.writeheader()
        for r in bookworm:
            writer.writerow({
                "id": r.id,
                "section_id": r.section_id,
                "section_title": r.section_title,
                "paragraph": r.paragraph,
                "is_hazard": "", "condition": "", "operation": "",
                "state": "", "hazard_type": "", "notes": "",
            })

    # Bookworm stats
    bw_hi = sum(1 for r in bookworm if any(w in r.paragraph.lower() for w in hi_words))
    bw_tables = sum(1 for r in bookworm if r.is_table)
    print(f"Sections: {', '.join(BOOKWORM_PREFIXES)}")
    print(f"Content paragraphs: {len(bookworm)}")
    print(f"Estimated HIs (keyword match): ~{bw_hi}")
    print(f"\nBy section:")
    for prefix in BOOKWORM_PREFIXES:
        paras = [r for r in bookworm if r.section_id.startswith(prefix)]
        hi_c = sum(1 for r in paras if any(w in r.paragraph.lower() for w in hi_words))
        print(f"  {prefix}: {len(paras)} paragraphs, ~{hi_c} HIs")
    print(f"\n✓ Saved Bookworm subset:")
    print(f"  {BOOKWORM_JSONL}")
    print(f"  {BOOKWORM_CSV}")


if __name__ == "__main__":
    main()