from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.generation.rag_llm import run_rag

app = FastAPI(
    title="SEBI Compliance RAG API",
    description="SEBI regulatory retrieval + answer generation service.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    user: str = Field(..., min_length=1)
    assistant: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question about SEBI regulations")
    history: List[ChatTurn] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    id: Optional[str] = None
    source_file: str
    chunk_index: int
    doc_type: Optional[str] = None
    text: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    evidence: List[EvidenceItem]
    mode: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


def _detect_mode(answer: str) -> str:
    if "Generation provider unavailable" in answer or "Failure detail:" in answer:
        return "fallback"
    return "llm"


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="sebi-rag-api",
        version=app.version,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="sebi-rag-api",
        version=app.version,
    )


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    history: List[Dict[str, str]] = [turn.model_dump() for turn in request.history]

    try:
        result: Dict[str, Any] = run_rag(request.question, history=history)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {exc}") from exc

    answer = str(result.get("answer", ""))
    evidence_raw = result.get("evidence", [])
    evidence: List[EvidenceItem] = []

    for item in evidence_raw:
        try:
            chunk_index = int(item.get("chunk_index", -1))
        except (TypeError, ValueError):
            chunk_index = -1

        score_raw = item.get("score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None

        evidence.append(
            EvidenceItem(
                id=item.get("id"),
                source_file=str(item.get("source_file", "unknown")),
                chunk_index=chunk_index,
                doc_type=item.get("doc_type"),
                text=str(item.get("text", "")),
                score=score,
            )
        )

    return QueryResponse(
        question=request.question,
        answer=answer,
        evidence=evidence,
        mode=_detect_mode(answer),
    )
