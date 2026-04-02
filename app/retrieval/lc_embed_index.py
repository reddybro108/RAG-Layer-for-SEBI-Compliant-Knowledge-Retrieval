# app/retrieval/lc_embed_index.py
"""
Build FAISS index using LangChain + HuggingFace embeddings.

Reads:  data/data_processed/corpus.jsonl
Writes: faiss_index/langchain_index/
"""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


CORPUS_PATH = Path("data/data_processed/corpus.jsonl")
INDEX_DIR = Path("faiss_index/langchain_index")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus file not found at {CORPUS_PATH}. Run your ingestion pipeline first."
        )

    texts = []
    metadatas = []

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metadatas.append(
                {
                    "id": obj.get("id"),
                    "source_file": obj.get("source_file"),
                    "chunk_index": obj.get("chunk_index"),
                    "doc_type": obj.get("doc_type"),
                }
            )

    return texts, metadatas


def build_langchain_faiss_index():
    print("Loading corpus from:", CORPUS_PATH)
    texts, metadatas = load_corpus()
    print(f"Total chunks: {len(texts)}")

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the embedding model. Ensure "
            f"'{EMBED_MODEL}' is available locally or allow network access to "
            "download it."
        ) from exc

    print("Building FAISS index via LangChain...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print("\nLangChain FAISS index built successfully.")
    print(f"Index directory: {INDEX_DIR}")


if __name__ == "__main__":
    build_langchain_faiss_index()
