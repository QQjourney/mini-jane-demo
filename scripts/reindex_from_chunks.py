# scripts/reindex_from_chunks.py
import os, json, sys
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(ROOT / "vectorstore"))
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "jenosize-ideas")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

RESET = True  # เซ็ต True เพื่อเคลียร์คอลเลกชันก่อน (ชัวร์สุด)

def make_id(it): 
    return f"{it['doc_id']}#chunk{it['chunk_index']}"

def batched(lst, size=64):
    buf = []
    for x in lst:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    assert CHUNKS_FILE.exists(), f"Not found: {CHUNKS_FILE}"
    print("CHUNKS_FILE       :", CHUNKS_FILE)
    print("CHROMA_PERSIST_DIR:", CHROMA_PERSIST_DIR)
    print("COLLECTION_NAME   :", COLLECTION_NAME)
    print("EMBED_MODEL       :", EMBED_MODEL)

    # โหลด chunks
    chunks = [json.loads(l) for l in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("Loaded chunks     :", len(chunks))

    # ล้างซ้ำในหน่วยความจำ (กัน ID ชน)
    uniq = {}
    for c in chunks:
        uniq[make_id(c)] = c
    chunks = list(uniq.values())
    print("After dedupe      :", len(chunks))

    # เตรียม Chroma
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(allow_reset=True))
    if RESET:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION_NAME)

    # โหลด embedder
    embedder = SentenceTransformer(EMBED_MODEL)

    # upsert
    added = 0
    seen = set()
    for batch in batched(chunks, size=64):
        ids, docs, metas = [], [], []
        for b in batch:
            cid = make_id(b)
            if cid in seen: 
                continue
            seen.add(cid)
            ids.append(cid)
            docs.append(b["text"])
            metas.append({
                "url": b["url"], "title": b["title"], "category": b["category"],
                "section": b["section"], "chunk_index": b["chunk_index"],
                "language": b.get("language","en")
            })
        if not ids:
            continue
        embs = embedder.encode(docs, normalize_embeddings=True).tolist()
        col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        added += len(ids)

    print("Upserted          :", added)
    try:
        print("Collection count  :", col.count())
    except Exception as e:
        print("count error:", e)

if __name__ == "__main__":
    sys.exit(main())
