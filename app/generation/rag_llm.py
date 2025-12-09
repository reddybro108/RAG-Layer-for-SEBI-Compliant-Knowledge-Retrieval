# app/generation/rag_llm.py

from typing import Optional, List, Dict, Any

from app.retrieval.lc_rag_engine import LangChainRAGEngine
from app.generation.groq_llm import groq_call


def run_rag(question: str, history: Optional[List[Dict[str, str]]] = None):
    """
    Orchestrates LangChain-based retrieval + prompt creation + Groq LLM answer generation.
    """
    history = history or []

    rag = LangChainRAGEngine(top_k=5)
    contexts = rag.retrieve(question)
    prompt = rag.build_prompt(question, contexts, history=history)

    answer = groq_call(prompt)

    return {
        "question": question,
        "answer": answer,
        "evidence": contexts,
    }


if __name__ == "__main__":
    out = run_rag("What are the disclosure requirements for listed entities?")
    print("ANSWER:\n", out["answer"])
    print("\nEVIDENCE:\n", out["evidence"])


# # generation/rag_llm.py

# import os
# import sys
# from pathlib import Path

# # --- Dynamic path injection so direct execution works -----------------
# ROOT_DIR = Path(__file__).resolve().parents[2]   # project root
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))
# # ----------------------------------------------------------------------

# from app.retrieval.rag_engine import RAGEngine
# from app.generation.groq_llm import groq_call


# def run_rag(question: str, history: list[str] | None = None):
#     """Orchestrates retrieval + prompt creation + LLM answer generation."""
    
#     rag = RAGEngine(top_k=5)
#     contexts = rag.retrieve(question)
#     prompt = rag.build_prompt(question, contexts, history=history)
#     answer = groq_call(prompt)

#     return {
#         "question": question,
#         "answer": answer,
#         "evidence": contexts,
#     }


# if __name__ == "__main__":
#     result = run_rag("What are the disclosure requirements for listed entities?")

#     print("\n===== ANSWER =====\n")
#     print(result["answer"])

#     print("\n===== EVIDENCE =====\n")
#     for item in result["evidence"]:
#         print(f"- {item['source_file']} (chunk {item['chunk_index']}) | score={item['score']}")
