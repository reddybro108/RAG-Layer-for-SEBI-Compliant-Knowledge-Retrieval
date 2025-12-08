# app/build_corpus.py

import json
import uuid
from pathlib import Path

PROCESSED_DIR = Path("data/data_processed")
CORPUS_FILE = PROCESSED_DIR / "corpus.jsonl"

CHUNK_SIZE = 1500
OVERLAP = 200

def chunk_text(text: str):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP

    return chunks

def build_corpus():
    json_files = list(PROCESSED_DIR.glob("*.json"))
    if not json_files:
        print("❌ No JSON files in data_processed/. Run parse_pdfs.py first.")
        return

    if CORPUS_FILE.exists():
        CORPUS_FILE.unlink()

    with CORPUS_FILE.open("w", encoding="utf-8") as f_out:
        for fp in json_files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            source = data["source_file"]
            text = data["raw_text"]

            chunks = chunk_text(text)

            for idx, chunk in enumerate(chunks):
                record = {
                    "id": str(uuid.uuid4()),
                    "source_file": source,
                    "chunk_index": idx,
                    "doc_type": source.split("_")[0],
                    "text": chunk,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[✓] Chunked: {source} → {len(chunks)} chunks")

    print(f"\n[🚀] Corpus created at: {CORPUS_FILE}")

if __name__ == "__main__":
    build_corpus()
