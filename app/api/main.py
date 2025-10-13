# app/api/main.py
from __future__ import annotations
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.schemas import (
    AskRequest, AskResponse, RetrievalItem,
    SearchRequest, SearchResponse, HealthResponse, CategoriesResponse
)
from app.rag.pipeline import answer
from app.rag.retriever import get_collection, search as retriever_search

load_dotenv()

APP_NAME = os.getenv("PROJECT_NAME", "mini-jane-demo")
ENV = os.getenv("APP_ENV", "development")

app = FastAPI(title=APP_NAME, version="0.1.0")

# --- CORS (เปิดกว้างช่วง dev; โปรดปรับโดเมนจริงตอนโปรดักชัน) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    res = answer(
        query=req.query,
        mode=req.mode,
        k=req.k,
        where=req.where,
        where_document=req.where_document,
    )
    # แปลงเป็น Pydantic model
    retrieval = [
        RetrievalItem(text=r["text"], distance=float(r["distance"]), meta=r["meta"])
        for r in res.get("retrieval", [])
    ]
    return AskResponse(
        answer=res["answer"],
        citations=res.get("citations", []),
        used_k=res.get("used_k", 0),
        retrieval=retrieval,
    )

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    results = retriever_search(
        req.query, k=req.k, where=req.where, where_document=req.where_document,
        use_mmr=True, fetch_k=max(25, req.k * 4)
    )
    out = [
        RetrievalItem(text=r["text"], distance=float(r["distance"]), meta=r["meta"])
        for r in results
    ]
    return SearchResponse(results=out)

@app.get("/categories", response_model=CategoriesResponse)
def categories():
    # ดึง categories จาก metadata ในคอลเลกชัน
    cats: List[str] = []
    try:
        col = get_collection()
        data = col.get(limit=2000)
        metas = data.get("metadatas") or []
        seen = set()
        for m in metas:
            c = (m or {}).get("category")
            if c and c not in seen:
                seen.add(c); cats.append(c)
        cats.sort()
    except Exception:
        cats = [
            "Futurist",
            "Understand People & Consumer",
            "Transformation & Technology",
            "Utility for Our World",
            "Real-time Marketing",
            "Experience the New World",
        ]
    return CategoriesResponse(categories=cats)
