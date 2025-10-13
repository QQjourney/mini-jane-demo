# app/rag/embeddings.py
from pathlib import Path
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def embed_texts(texts, normalize=True):
    emb = get_embedder().encode(texts, normalize_embeddings=normalize).tolist()
    return emb

def embed_query(text):
    return embed_texts([text], normalize=True)[0]
