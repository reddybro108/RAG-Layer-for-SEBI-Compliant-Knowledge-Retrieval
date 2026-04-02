# app/retrieval/lc_rag_engine.py

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import disable_broken_local_proxy, get_embed_model

INDEX_DIR = Path("faiss_index/langchain_index")
EMBED_MODEL = get_embed_model()


class LangChainRAGEngine:
    """RAG engine using LangChain's FAISS VectorStore and HuggingFace embeddings."""

    def __init__(self, top_k: int = 5):
        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"LangChain FAISS index not found at {INDEX_DIR}. "
                "Run: python -m app.retrieval.lc_embed_index"
            )

        print("Initializing LangChain RAG engine...")

        self.top_k = top_k
        disable_broken_local_proxy()
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the embedding model. Ensure "
                f"'{EMBED_MODEL}' is available locally or allow network access to "
                "download it."
            ) from exc
        self.vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

        print("LangChain RAG engine ready.")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Return top-k relevant chunks along with relevance scores."""
        print(f"\nRetrieving top {self.top_k} matches for query:")
        print(f"   -> {query}")

        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=self.top_k,
        )

        results = []
        for doc, score in docs_and_scores:
            meta = dict(doc.metadata or {})
            meta["text"] = doc.page_content
            meta["score"] = float(score)
            results.append(meta)

        return results

    def build_prompt(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build a SEBI-compliant prompt using retrieved context and history."""
        history_text = ""

        if history:
            for turn in history:
                history_text += (
                    f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
                )

        context_block = "\n\n".join(
            f"[{c.get('source_file')} - chunk {c.get('chunk_index')}]:\n{c.get('text')}"
            for c in contexts
        )

        prompt = f"""
You are a SEBI compliance assistant.
Answer strictly from the retrieved SEBI context.
Use only the most relevant point from the provided context and do not include unrelated details.

If the answer is not clearly supported by the context, reply exactly:
"The referenced SEBI documents do not cover this information."

Conversation History:
{history_text}

User Question:
{query}

Retrieved SEBI Context:
{context_block}

Return 3-6 bullet points with precise compliance language and include one inline citation in this format:
[source_file - chunk chunk_index]
"""
        return prompt
