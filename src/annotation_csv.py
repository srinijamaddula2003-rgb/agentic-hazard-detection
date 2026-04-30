from pathlib import Path
import json
import csv

IN_PATH = Path("data/processed/TS_24.301_paragraphs.jsonl")
OUT_PATH = Path("data/processed/annotation_sample.csv")

def load_paragraphs(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records

if __name__ == "__main__":
    records = load_paragraphs(IN_PATH)
    print(f"Loaded {len(records)} paragraphs.")

    # Filter out table paragraphs — they are almost never hazard indicators
    content_records = [r for r in records if not r.get("is_table", False)]
    print(f"Content paragraphs (excluding tables): {len(content_records)}")

    fieldnames = [
        "id",
        "section_id",
        "section_title",
        "paragraph",
        "is_hazard",
        "condition",
        "operation",
        "state",
        "hazard_type",
        "notes",
    ]

    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, rec in enumerate(content_records):
            writer.writerow({
                "id": i,
                "section_id": rec.get("section_id", ""),
                "section_title": rec.get("section_title", ""),
                "paragraph": rec.get("paragraph", "").replace("\n", " "),
                "is_hazard": "",
                "condition": "",
                "operation": "",
                "state": "",
                "hazard_type": "",
                "notes": "",
            })

    print(f"Wrote {len(content_records)} rows to {OUT_PATH}")
    print("Now open this CSV in Excel and annotate every row.")