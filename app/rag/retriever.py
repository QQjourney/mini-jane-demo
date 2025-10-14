# app/rag/retriever.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import os
import numpy as np

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# ใช้ Embedding API แทน local model
from app.rag.embeddings_api import embed_query as _embed_query

load_dotenv()

# -------- ENV / PATH --------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./notebooks/vectorstore")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "jenosize-ideas")

# -------- Singleton clients --------
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Any] = None


def get_collection():
    """
    คืนค่าคอลเลกชันของ Chroma ที่ persist ลงโฟลเดอร์ CHROMA_PERSIST_DIR
    """
    global _client, _collection
    if _collection is None:
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(allow_reset=True)
        )
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


# --------- Similarity / MMR utilities ---------
def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def mmr(
    query_emb: List[float],
    doc_embs: List[List[float]],
    k: int = 5,
    lambda_mult: float = 0.6,
) -> List[int]:
    """
    Maximal Marginal Relevance: เลือกเอกสารที่ทั้ง 'เกี่ยวข้อง' และ 'หลากหลาย'
    - query_emb: เวกเตอร์ของ query
    - doc_embs : เวกเตอร์เอกสารที่ได้มาจาก Chroma (include=['embeddings'])
    - k        : จำนวนผลลัพธ์สุดท้าย
    - lambda_mult: ค่าถ่วงน้ำหนักระหว่าง relevance vs diversity
    Return: index ของเอกสารที่ถูกเลือก (เรียงตามคะแนน MMR)
    """
    n = len(doc_embs)
    if n == 0:
        return []

    rel = np.array([cosine(query_emb, e) for e in doc_embs], dtype=np.float32)

    selected = [int(np.argmax(rel))]
    candidates = set(range(n)) - set(selected)

    while len(selected) < min(k, n) and candidates:
        best_idx, best_val = None, -1e9
        for c in candidates:
            # ความหลากหลาย: document c คล้ายกับตัวที่เลือกไปแล้วมากแค่ไหน
            div = max(cosine(doc_embs[c], doc_embs[s]) for s in selected)
            score = lambda_mult * rel[c] - (1.0 - lambda_mult) * div
            if score > best_val:
                best_val, best_idx = score, c
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


# --------- Public API: search() ---------
def search(
    query: str,
    k: int = 5,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    use_mmr: bool = True,
    fetch_k: int = 25,
) -> List[Dict]:
    """
    ค้นหาชิ้นความรู้ที่เกี่ยวข้องจาก Chroma (RAG retriever)
    - query: คำถาม (TH/EN)
    - k: จำนวนผลลัพธ์สุดท้ายที่ต้องการ
    - where: ฟิลเตอร์ metadata (เช่น {"category": "Futurist"})
    - where_document: ฟิลเตอร์เนื้อเอกสาร (regex/fulltext ของ Chroma)
    - use_mmr: ใช้ MMR เพื่อเพิ่มความหลากหลายของเอกสาร
    - fetch_k: ดึงผลดิบชุดใหญ่ก่อน แล้วจึงคัดด้วย MMR เหลือ k
    Return: list ของ dict {text, meta, distance, similarity}
    หมายเหตุ: Chroma คืน 'distance' แบบ cosine distance (ยิ่งน้อยยิ่งใกล้)
    """
    col = get_collection()

    # สร้างเวกเตอร์คำถามด้วย Gemini Embedding API
    q_emb = _embed_query(query)

    # ยิงค้นหาเบื้องต้น
    res = col.query(
        query_embeddings=[q_emb],
        n_results=(fetch_k if use_mmr else k),
        include=["documents", "metadatas", "distances", "embeddings"],
        where=where,
        where_document=where_document,
    )

    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    embs  = res.get("embeddings", [[]])[0]

    if not docs:
        return []

    # คัดด้วย MMR หรือเอา top-k ตรง ๆ
    if use_mmr:
        idxs = mmr(q_emb, embs, k=k, lambda_mult=0.6)
    else:
        idxs = list(range(min(k, len(docs))))

    results: List[Dict] = []
    for i in idxs:
        dist = float(dists[i])
        # Chroma (cosine distance): similarity โดยประมาณ = 1 - distance
        sim  = 1.0 - dist
        results.append(
            {
                "text": docs[i],
                "meta": metas[i],
                "distance": dist,
                "similarity": sim,
            }
        )
    return results


# --------- Helper (optional) ---------
def build_where(
    category: Optional[str] = None,
    section: Optional[str] = None,
    language: Optional[str] = None,
) -> Optional[Dict]:
    """
    สร้าง where filter เบื้องต้นจากพารามิเตอร์ทั่วไป
    """
    cond: Dict[str, Any] = {}
    if category:
        cond["category"] = category
    if section:
        cond["section"] = section
    if language:
        cond["language"] = language
    return cond or None
