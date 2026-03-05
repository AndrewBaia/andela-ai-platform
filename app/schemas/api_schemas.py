from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the core responsibilities of an AI Engineer at Andela?")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    latency_ms: float

class IngestionResponse(BaseModel):
    status: str
    message: str
