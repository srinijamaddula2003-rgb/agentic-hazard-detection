from pathlib import Path
import pdfplumber
import re
from dataclasses import dataclass, asdict, field
import json

RAW_DIR = Path("data/raw_specs")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Matches real section headers like "5.4.4  EPS attach procedure"
# but NOT TOC lines (which have dotted leaders or page numbers at end)
SECTION_PATTERN = re.compile(r"^(\d+(\.\d+)*)\s+(.+)$")

# Page header/footer noise
PAGE_HEADER_PATTERN = re.compile(
    r"^(3GPP\s*$|Release\s+\d+\s+\d+\s+3GPP|3GPP\s+TS\s+24\.301)",
    re.IGNORECASE
)

@dataclass
class ParagraphRecord:
    spec: str
    section_id: str
    section_title: str
    paragraph: str
    is_table: bool = False  # tag table paragraphs so you can filter later

def pdf_to_text(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            parts.append(text)
    return "\n".join(parts)

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
    return False

def is_table_content(paragraph: str) -> bool:
    """Detect message format tables (IEI tables, etc.)"""
    indicators = [
        'IEI' in paragraph and 'Information Element' in paragraph,
        'Presence' in paragraph and 'Format' in paragraph and 'Length' in paragraph,
        paragraph.count('\t') > 3,  # tab-heavy = likely table
    ]
    return any(indicators)

def parse_sections(raw_text: str, spec_name: str = "TS 24.301"):
    records = []
    current_section_id = ""
    current_section_title = ""
    current_para_lines = []
    started = False  # skip everything before Section 1

    lines = raw_text.splitlines()
    for line in lines:
        line_stripped = line.strip()

        # Skip until we reach "1 Scope" (the real start of the spec)
        if not started:
            if re.match(r'^1\s+Scope', line_stripped) and not is_toc_line(line_stripped):
                started = True
                current_section_id = "1"
                current_section_title = "Scope"
            continue

        # Skip blank lines (paragraph break)
        if not line_stripped:
            if current_para_lines:
                paragraph_text = " ".join(current_para_lines).strip()
                if len(paragraph_text) >= 20:  # skip very short garbage
                    records.append(
                        ParagraphRecord(
                            spec=spec_name,
                            section_id=current_section_id,
                            section_title=current_section_title,
                            paragraph=paragraph_text,
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
                if len(paragraph_text) >= 20:
                    records.append(
                        ParagraphRecord(
                            spec=spec_name,
                            section_id=current_section_id,
                            section_title=current_section_title,
                            paragraph=paragraph_text,
                            is_table=is_table_content(paragraph_text),
                        )
                    )
                current_para_lines = []

            candidate_title = m.group(3).strip()
            # Make sure this isn't a TOC line that slipped through
            if not is_toc_line(candidate_title):
                current_section_id = m.group(1)
                current_section_title = candidate_title
        else:
            current_para_lines.append(line_stripped)

    # Flush last paragraph
    if current_para_lines:
        paragraph_text = " ".join(current_para_lines).strip()
        if len(paragraph_text) >= 20:
            records.append(
                ParagraphRecord(
                    spec=spec_name,
                    section_id=current_section_id,
                    section_title=current_section_title,
                    paragraph=paragraph_text,
                    is_table=is_table_content(paragraph_text),
                )
            )

    return records

if __name__ == "__main__":
    pdf_path = RAW_DIR / "TS_24.301.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Could not find PDF at {pdf_path}")

    raw_text = pdf_to_text(pdf_path)
    records = parse_sections(raw_text, spec_name="TS 24.301")
    print(f"Parsed {len(records)} paragraphs.")

    # Show stats
    tables = sum(1 for r in records if r.is_table)
    non_tables = len(records) - tables
    print(f"  Content paragraphs: {non_tables}")
    print(f"  Table paragraphs: {tables}")

    # Spot check: print first 5 records
    for r in records[:5]:
        print(f"  [{r.section_id}] {r.section_title[:40]} | {r.paragraph[:80]}...")

    out_path = OUT_DIR / "TS_24.301_paragraphs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_path}")