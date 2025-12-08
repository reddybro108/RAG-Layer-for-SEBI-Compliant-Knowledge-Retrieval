# RAG/rag_engine.py

import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("faiss_index/faiss_index.bin")
META_PATH = Path("faiss_index/metadata.parquet")

class RAGEngine:

    def __init__(self, top_k: int = 5):
        if not INDEX_PATH.exists():
            raise FileNotFoundError("FAISS index not found. Run embed_index.py first.")
        if not META_PATH.exists():
            raise FileNotFoundError("Metadata file not found.")

        print("🚀 Initializing RAG engine...")
        self.top_k = top_k
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.meta = pd.read_parquet(META_PATH)
        print("✅ RAG Engine Ready.")

    def retrieve(self, query: str):
        print(f"\n🔎 Retrieving top {self.top_k} matches for query:")
        print(f"   → {query}")

        q_emb = self.model.encode([query])
        q_emb = np.asarray(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)

        scores, idxs = self.index.search(q_emb, self.top_k)
        scores = scores[0]
        idxs = idxs[0]

        results = []
        for idx, score in zip(idxs, scores):
            rec = self.meta.iloc[idx].to_dict()
            rec["score"] = float(score)
            results.append(rec)

        return results

    def build_prompt(self, query: str, contexts, history=None):
        """Build SEBI-compliant RAG prompt with optional user history."""

        history_text = ""
        if history:
            for i, q in enumerate(history, start=1):
                history_text += f"User({i}): {q}\n"

        context_block = "\n\n".join(
            f"[{c['source_file']} - chunk {c['chunk_index']}]:\n{c['text']}"
            for c in contexts
        )

        prompt = f"""
You are a SEBI compliance assistant.
Answer strictly using the retrieved SEBI context below.
If the answer is NOT present in the context, respond:
"The referenced SEBI documents do not cover this information."

Conversation History:
{history_text}

User Question:
{query}

Retrieved SEBI Context:
{context_block}

Provide a concise, regulation-accurate answer with citations to source_file and chunk_index.
"""
        return prompt
