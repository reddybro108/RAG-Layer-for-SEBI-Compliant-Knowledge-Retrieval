# app/api_server.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_llm import run_rag

app = FastAPI(
    title="SEBI Compliance RAG API",
    description="Retrieval-Augmented Generation backend for SEBI regulations.",
    version="1.0.0"
)

# Request schema
class QueryRequest(BaseModel):
    question: str

# Response schema
class QueryResponse(BaseModel):
    question: str
    answer: str
    evidence: list

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = run_rag(request.question)
    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        evidence=result["evidence"]
    )
