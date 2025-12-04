# app/embed_index.py

import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CORPUS_FILE = Path("data_processed/corpus.jsonl")
INDEX_PATH = Path("data_processed/faiss_index.bin")
META_PATH = Path("data_processed/metadata.parquet")

def build_index():
    if not CORPUS_FILE.exists():
        print("❌ corpus.jsonl not found. Run build_corpus.py first.")
        return

    print("🚀 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, high quality

    texts = []
    metadata = []

    print("📄 Reading corpus...")
    with CORPUS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metadata.append(rec)

    print(f"📦 Total chunks: {len(texts)}")
    print("⚙️  Generating embeddings...")

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Normalized for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine-ready)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, str(INDEX_PATH))
    pd.DataFrame(metadata).to_parquet(META_PATH, index=False)

    print("\n🎯 FAISS index built successfully!")
    print(f"📌 Index saved at: {INDEX_PATH}")
    print(f"📌 Metadata saved at: {META_PATH}")

if __name__ == "__main__":
    build_index()
