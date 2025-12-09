# app/retrieval/lc_rag_engine.py

from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


INDEX_DIR = Path("faiss_index/langchain_index")


class LangChainRAGEngine:
    """
    RAG engine using LangChain's FAISS VectorStore and HuggingFace embeddings.
    """

    def __init__(self, top_k: int = 5):
        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"LangChain FAISS index not found at {INDEX_DIR}. "
                "Run: python -m app.retrieval.embed_index"
            )

        print("🚀 Initializing LangChain RAG engine...")

        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # allow_dangerous_deserialization is required in newer LangChain versions
        self.vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        print("✅ LangChain RAG Engine Ready.")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Return top-k relevant chunks using LangChain retriever."""
        print(f"\n🔎 Retrieving top {self.top_k} matches for query:")
        print(f"   → {query}")

        docs = self.retriever.invoke(query)

        results = []
        for doc in docs:
            meta = doc.metadata or {}
            meta["text"] = doc.page_content
            # Optional: add a dummy score, FAISS distance not directly exposed here
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
                history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        context_block = "\n\n".join(
            f"[{c.get('source_file')} - chunk {c.get('chunk_index')}]:\n{c.get('text')}"
            for c in contexts
        )

        prompt = f"""
You are a SEBI compliance assistant. 
You must answer strictly using the context provided from SEBI regulations and circulars.

If the information is NOT present in the retrieved SEBI documents, reply:
"The referenced SEBI documents do not cover this information."

Conversation History:
{history_text}

User Question:
{query}

Retrieved SEBI Context:
{context_block}

Provide a concise, compliant answer with inline references to the source_file and chunk_index where relevant.
"""
        return prompt
