# api/api_server.py

from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.generation.rag_llm import run_rag

app = FastAPI(
    title="SEBI Compliance RAG API",
    description="RAG backend for SEBI regulations.",
    version="1.0.0",
)

class ChatTurn(BaseModel):
    user: str = Field(..., min_length=1)
    assistant: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatTurn]] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    evidence: list

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    history = [turn.model_dump() for turn in request.history or []]
    result = run_rag(request.question, history=history)
    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        evidence=result["evidence"],
    )
