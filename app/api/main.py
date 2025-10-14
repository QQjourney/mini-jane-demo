# app/api/main.py
from __future__ import annotations

import os
import json
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
from app.rag.embeddings import embed_docs  # ใช้ Gemini text-embedding-004

# ===== Load envs =====
load_dotenv()

APP_NAME   = os.getenv("PROJECT_NAME", "mini-jane-demo")
APP_ENV    = os.getenv("APP_ENV", "development")
DATA_PATH  = os.getenv("DATA_PATH", "./data")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
COLL_NAME  = os.getenv("COLLECTION_NAME", "jenosize-ideas")
AUTO_REINDEX = os.getenv("AUTO_REINDEX", "1")  # "1" = เปิดอัตโนมัติเมื่อว่าง

# ===== FastAPI app =====
app = FastAPI(title=APP_NAME, version="0.1.0")

# CORS (เปิดกว้างช่วง dev; โปรดกำหนดโดเมนจริงตอน prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Helpers ----------
def auto_reindex_from_chunks() -> None:
    """
    สร้างเวกเตอร์สโตร์จาก data/processed/chunks.jsonl ถ้าคอลเลกชันว่าง
    ใช้ได้ทั้ง Render (CHROMA_DIR=/tmp/vectorstore) และ local
    """
    try:
        col = get_collection()
        existing = 0
        try:
            existing = col.count()
        except Exception:
            existing = 0

        if existing and existing > 0:
            print(f"✅ Vectorstore ready: {existing} chunks @ {CHROMA_DIR} (collection='{COLL_NAME}')")
            return

        chunks_file = os.path.join(DATA_PATH, "processed", "chunks.jsonl")
        if not os.path.exists(chunks_file):
            print(f"⚠️ Not found: {chunks_file}  -> skip auto-index")
            return

        print(f"⚙️ Building vectorstore from {chunks_file} ...")
        rows = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            print("⚠️ chunks.jsonl is empty, skip.")
            return

        docs  = [r["text"] for r in rows]
        metas = [r["meta"] for r in rows]
        ids   = [f"doc-{i}" for i in range(len(docs))]

        # ฝังเวกเตอร์ด้วย Gemini Embedding API (models/text-embedding-004)
        embs  = embed_docs(docs)
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        print(f"✅ Indexed {len(docs)} chunks -> {CHROMA_DIR} (collection='{COLL_NAME}')")
    except Exception as e:
        print(f"❌ auto_reindex failed: {e}")

# ---------- Lifecycle ----------
@app.on_event("startup")
def on_startup():
    print("=== Startup Info ===")
    print(f"ENV: {APP_ENV}")
    print(f"CHROMA_PERSIST_DIR: {CHROMA_DIR}")
    print(f"COLLECTION_NAME   : {COLL_NAME}")
    print(f"AUTO_REINDEX      : {AUTO_REINDEX}")
    if AUTO_REINDEX == "1":
        auto_reindex_from_chunks()

# ---------- Basic endpoints ----------
@app.get("/", tags=["meta"])
def root():
    return {"message": "Mini Jane API is live!"}

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return {"status": "ok"}

# ---------- RAG endpoints ----------
@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest):
    res = answer(
        query=req.query,
        mode=req.mode,
        k=req.k,
        where=req.where,
        where_document=req.where_document,
    )
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

@app.post("/search", response_model=SearchResponse, tags=["rag"])
def search(req: SearchRequest):
    results = retriever_search(
        req.query,
        k=req.k,
        where=req.where,
        where_document=req.where_document,
        use_mmr=True,
        fetch_k=max(25, req.k * 4),
    )
    out = [
        RetrievalItem(text=r["text"], distance=float(r["distance"]), meta=r["meta"])
        for r in results
    ]
    return SearchResponse(results=out)

@app.get("/categories", response_model=CategoriesResponse, tags=["rag"])
def categories():
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
