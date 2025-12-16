from pydantic import BaseModel
from typing import List
from datetime import datetime

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    section: str
    version: str
    created_at: datetime

class RetrievedChunk(BaseModel):
    content: str
    metadata: ChunkMetadata
    score: float

class RetrievalResponse(BaseModel):
    query: str
    top_k: int
    results: List[RetrievedChunk]
