# app/rag/retriever.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os, re
import numpy as np
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from .embeddings import embed_query

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "jenosize-ideas")

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(allow_reset=True))
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection

def cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def mmr(query_emb, doc_embs, k=5, lambda_mult=0.6):
    n = len(doc_embs)
    if n == 0: return []
    rel = np.array([cosine(query_emb, e) for e in doc_embs])
    selected = [int(np.argmax(rel))]
    candidates = set(range(n)) - set(selected)
    while len(selected) < min(k, n) and candidates:
        best, best_val = None, -1e9
        for c in candidates:
            div = max(cosine(doc_embs[c], doc_embs[s]) for s in selected)
            score = lambda_mult*rel[c] - (1-lambda_mult)*div
            if score > best_val:
                best_val, best = score, c
        selected.append(best); candidates.remove(best)
    return selected

def search(
    query: str,
    k: int = 5,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    use_mmr: bool = True,
    fetch_k: int = 25,
):
    col = get_collection()
    q_emb = embed_query(query)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=fetch_k if use_mmr else k,
        include=["documents", "metadatas", "distances", "embeddings"],
        where=where,
        where_document=where_document
    )
    docs   = res.get("documents", [[]])[0]
    metas  = res.get("metadatas", [[]])[0]
    dists  = res.get("distances", [[]])[0]
    embs   = res.get("embeddings", [[]])[0]

    if not docs:
        return []

    if use_mmr:
        idxs = mmr(q_emb, embs, k=k, lambda_mult=0.6)
        out = [(docs[i], metas[i], dists[i]) for i in idxs]
    else:
        out = list(zip(docs[:k], metas[:k], dists[:k]))

    # ใส่ score (similarity) ไว้ใน meta_debug
    results = []
    for doc, meta, dist in out:
        results.append({
            "text": doc,
            "meta": meta,
            "distance": float(dist)
        })
    return results
