# app/api/schemas.py
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field("RAG", pattern="^(RAG|LLM)$")
    k: int = 5
    where: Optional[Dict] = None
    where_document: Optional[Dict] = None

class RetrievalItem(BaseModel):
    text: str
    distance: float
    meta: Dict

class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    used_k: int
    retrieval: List[RetrievalItem]

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    where: Optional[Dict] = None
    where_document: Optional[Dict] = None

class SearchResponse(BaseModel):
    results: List[RetrievalItem]

class HealthResponse(BaseModel):
    status: str

class CategoriesResponse(BaseModel):
    categories: List[str]
