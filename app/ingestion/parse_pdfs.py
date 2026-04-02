# app/ingestion/parse_pdfs.py

import json
from pathlib import Path

import fitz

RAW_DIR = Path("data/data_raw")
OUT_DIR = Path("data/data_processed")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n\n".join(pages)


def process_pdfs():
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/data_raw/")
        return

    for pdf in pdf_files:
        text = extract_text(pdf)
        payload = {
            "source_file": pdf.name,
            "raw_text": text,
        }

        out_file = OUT_DIR / f"{pdf.stem}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Parsed: {pdf.name}")


if __name__ == "__main__":
    process_pdfs()
