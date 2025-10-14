# app/rag/retriever.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# ฝั่ง embedding (Gemini text-embedding-004) ต้องมีฟังก์ชันนี้อยู่แล้ว
from .embeddings import embed_query

load_dotenv()
log = logging.getLogger(__name__)

# ---- ENV & Defaults ----
# default ให้ตรงโครงสร้าง repo (ใช้บน Render/ลินุกซ์ได้)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./notebooks/vectorstore")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "jenosize-ideas")

# singletons
_client = None
_collection = None


# ---- Utils ----
def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


def _mmr(query_emb: List[float], doc_embs: List[List[float]], k: int = 5, lambda_mult: float = 0.6) -> List[int]:
    """Maximal Marginal Relevance: เลือกเอกสารให้หลากหลาย + เกี่ยวข้อง"""
    n = len(doc_embs)
    if n == 0:
        return []
    rel = np.array([_cosine(query_emb, e) for e in doc_embs])
    selected = [int(np.argmax(rel))]
    candidates = set(range(n)) - set(selected)

    while len(selected) < min(k, n) and candidates:
        best, best_val = None, -1e9
        for c in candidates:
            div = max(_cosine(doc_embs[c], doc_embs[s]) for s in selected)
            score = lambda_mult * rel[c] - (1 - lambda_mult) * div
            if score > best_val:
                best_val, best = score, c
        selected.append(best)
        candidates.remove(best)
    return selected


# ---- Chroma client / collection ----
def get_collection():
    """คืนค่า Chroma collection (lazy init + log บอกสถานะชัดเจน)"""
    global _client, _collection

    if _collection is None:
        # make sure path exists (กรณี AUTO_REINDEX=0 และเรามีไฟล์ฝั่ง repo)
        p = Path(CHROMA_PERSIST_DIR)
        p.mkdir(parents=True, exist_ok=True)

        log.info(f"[RAG] CHROMA_PERSIST_DIR={p.resolve()}")
        log.info(f"[RAG] COLLECTION_NAME={COLLECTION_NAME}")

        # แนบรายละเอียดไฟล์เล็กน้อยเพื่อ debug บน Render
        try:
            sample = [f.name for f in p.glob("*")][:20]
            log.info(f"[RAG] Vectorstore files sample: {sample}")
        except Exception as e:
            log.warning(f"[RAG] list vectorstore files failed: {e}")

        # สร้าง persistent client + collection
        _client = chromadb.PersistentClient(
            path=str(p),
            settings=Settings(allow_reset=True),
        )
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)

        # log count
        try:
            cnt = _collection.count()
            log.info(f"[RAG] Collection ready -> {COLLECTION_NAME} | count={cnt}")
        except Exception as e:
            log.exception(f"[RAG] Failed to count collection: {e}")

    return _collection


# ---- Search API ----
def search(
    query: str,
    k: int = 5,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    use_mmr: bool = True,
    fetch_k: int = 25,
) -> List[Dict]:
    """
    คืนรายการเอกสารรูปแบบ:
    [
      { "text": str, "meta": dict, "distance": float },
      ...
    ]
    """
    col = get_collection()
    # ฝังเวกเตอร์คิวรีด้วย Gemini (768d)
    q_emb = embed_query(query)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=fetch_k if use_mmr else k,
        include=["documents", "metadatas", "distances", "embeddings"],
        where=where,
        where_document=where_document,
    )

    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    embs  = res.get("embeddings", [[]])[0]  # บาง backend อาจคืน None; ถ้าไม่มีให้ปิด use_mmr

    if not docs:
        return []

    if use_mmr and embs:
        idxs = _mmr(q_emb, embs, k=k, lambda_mult=0.6)
        picked = [(docs[i], metas[i], dists[i]) for i in idxs]
    else:
        picked = list(zip(docs[:k], metas[:k], dists[:k]))

    results = []
    for doc, meta, dist in picked:
        results.append({
            "text": doc,
            "meta": meta or {},
            "distance": float(dist),
        })
    return results
