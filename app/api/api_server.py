# api/api_server.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from app.generation.rag_llm import run_rag

app = FastAPI(
    title="SEBI Compliance RAG API",
    description="RAG backend for SEBI regulations.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[str]] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    evidence: list

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = run_rag(request.question, history=request.history or [])
    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        evidence=result["evidence"],
    )
