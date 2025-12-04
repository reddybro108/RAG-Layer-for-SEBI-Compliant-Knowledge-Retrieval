# app/rag_llm.py

from app.rag_engine import RAGEngine
from app.groq_llm import groq_call

def run_rag(query: str):
    rag = RAGEngine(top_k=5)

    contexts = rag.retrieve(query)
    prompt = rag.build_prompt(query, contexts)

    # Now uses llama3-70b-8192 by default
    answer = groq_call(prompt)

    return {
        "query": query,
        "answer": answer,
        "evidence": contexts
    }

if __name__ == "__main__":
    out = run_rag("What are the disclosure requirements for listed entities?")
    print("\nANSWER:\n", out["answer"])
    print("\nEVIDENCE:\n", out["evidence"])
