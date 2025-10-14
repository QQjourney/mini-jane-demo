# app/ui/streamlit_app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os, json, time
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ------- RAG imports -------
from app.rag.pipeline import answer
from app.rag.retriever import get_collection
from app.rag.embeddings_api import embed_docs  # ใช้ตอน reindex

# ------- Page config -------
st.set_page_config(page_title="mini-jane-demo", page_icon="🤖", layout="wide")

# ------- ENV / PATH -------
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./vectorstore"))
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "jenosize-ideas")
CHUNKS_FILE        = Path(os.getenv("CHUNKS_FILE", "./data/processed/chunks.jsonl"))
GEN_MODEL          = os.getenv("GENERATION_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL_API    = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

# =========================
# Helpers
# =========================
def build_where_from_filters(selected_cats: List[str]) -> Optional[Dict]:
    if not selected_cats:
        return None
    if len(selected_cats) == 1:
        return {"category": selected_cats[0]}
    return {"category": {"$in": selected_cats}}

def ensure_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def get_all_categories(limit: int = 2000) -> List[str]:
    """อ่าน metadata บางส่วนจาก Chroma เพื่อดึงรายชื่อ category"""
    try:
        col = get_collection()
        res = col.get(limit=limit)
        metas = res.get("metadatas", []) or []
        cats = {m.get("category", "-") for m in metas if isinstance(m, dict)}
        cats.discard("-")
        items = sorted([c for c in cats if c])
        return items or [
            "Futurist",
            "Understand People & Consumer",
            "Transformation & Technology",
            "Utility for Our World",
            "Real-time Marketing",
            "Experience the New World",
        ]
    except Exception:
        return [
            "Futurist",
            "Understand People & Consumer",
            "Transformation & Technology",
            "Utility for Our World",
            "Real-time Marketing",
            "Experience the New World",
        ]

# =========================
# Auto-Index (first run)
# =========================
LOCK = CHROMA_PERSIST_DIR / ".building.lock"

def make_id(url: str, chunk_index: int) -> str:
    return f"{url}#chunk{int(chunk_index)}"

def _load_chunks() -> List[Dict]:
    assert CHUNKS_FILE.exists(), f"Not found: {CHUNKS_FILE}"
    out = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({
                "url": r.get("url",""),
                "title": r.get("title",""),
                "category": r.get("category",""),
                "section": r.get("section",""),
                "chunk_index": int(r.get("chunk_index", 0)),
                "text": r.get("text",""),
                "language": r.get("language","en")
            })
    return out

def ensure_index_ready():
    """ถ้ายังไม่มี index ให้สร้างจาก chunks.jsonl อัตโนมัติ (เหมาะกับ Render/Cloud)"""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    col = get_collection()

    try:
        n = col.count()
        if n and n > 0:
            st.sidebar.info(f"🔎 Vector index ready: {n} chunks")
            return
    except Exception:
        pass

    if not CHUNKS_FILE.exists():
        st.sidebar.error("❌ Missing chunks.jsonl – cannot build index.")
        return

    # กันชนพร้อมกันหลายคนด้วย lock ง่าย ๆ
    if LOCK.exists():
        st.sidebar.info("⏳ Another session is building index…")
        for _ in range(60):
            time.sleep(2)
            try:
                if col.count() > 0:
                    st.sidebar.success("✅ Index ready")
                    return
            except Exception:
                pass
        st.sidebar.error("⚠️ Index still not ready. Please refresh.")
        return

    try:
        LOCK.write_text("building", encoding="utf-8")
        st.sidebar.warning("⚙️ Building vector index (first run). Please wait…")

        raw = _load_chunks()
        # de-dup (url, chunk_index)
        seen = set()
        chunks = []
        for r in raw:
            key = (r["url"], r["chunk_index"])
            if key in seen:
                continue
            seen.add(key)
            chunks.append(r)

        BATCH = 64
        total, added = len(chunks), 0
        pbar = st.sidebar.progress(0.0, text="Indexing…")

        for i in range(0, total, BATCH):
            batch = chunks[i:i+BATCH]
            ids   = [make_id(b["url"], b["chunk_index"]) for b in batch]
            docs  = [b["text"] for b in batch]
            metas = [{
                "url": b["url"], "title": b["title"], "category": b["category"],
                "section": b["section"], "chunk_index": b["chunk_index"],
                "language": b.get("language","en")
            } for b in batch]

            embs = embed_docs(docs)  # ✅ Gemini Embedding API
            col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
            added += len(batch)
            pbar.progress(min(1.0, added/total), text=f"Indexing… {added}/{total}")

        pbar.empty()
        st.sidebar.success(f"✅ Index built: {added} chunks")
    finally:
        if LOCK.exists():
            LOCK.unlink(missing_ok=True)

# ===== Build UI =====
st.sidebar.title("⚙️ Settings")

# ปุ่ม rebuild สำหรับผู้ดูแล (จะลบทั้งคอลเลกชันแล้วสร้างใหม่)
with st.sidebar.expander("Admin • Maintenance"):
    if st.button("🧹 Rebuild index (danger)"):
        try:
            col = get_collection()
            # ล้างทั้งคอลเลกชัน
            from chromadb.config import Settings
            client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR), settings=Settings(allow_reset=True))
            client.delete_collection(COLLECTION_NAME)
            # สร้างใหม่ + สร้างดัชนี
            get_collection()  # โหลดใหม่
            ensure_index_ready()
            st.success("Rebuilt successfully.")
        except Exception as e:
            st.error(f"Rebuild failed: {e}")

mode  = st.sidebar.radio("Mode", options=["📚 RAG", "💬 LLM only"], index=0)
top_k = st.sidebar.slider("Top-K documents", min_value=3, max_value=10, value=5, step=1)

# ทำให้แน่ใจว่ามีดัชนีพร้อมใช้งาน
ensure_index_ready()

available_cats = get_all_categories()
selected_cats  = st.sidebar.multiselect("Filter by category (optional)", options=available_cats, default=[])

show_debug = st.sidebar.checkbox("Show retriever debug", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Vector store: `{CHROMA_PERSIST_DIR}`")
st.sidebar.caption(f"Collection : `{COLLECTION_NAME}`")
st.sidebar.caption(f"Embed (API): `{EMBED_MODEL_API}`")
st.sidebar.caption(f"Gen model  : `{GEN_MODEL}`")
st.sidebar.markdown("---")
st.sidebar.caption("Tip: พิมพ์ได้ทั้งไทย/อังกฤษ ระบบจะตอบตามภาษา")

# -------- Main Chat UI --------
st.title("mini-jane-demo 🤖")
st.caption("Chat with RAG over Jenosize Ideas (bilingual TH/EN) — with citations.")

ensure_session()

# แสดงประวัติแชต
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# รับคำถาม
query = st.chat_input("ถามคำถามได้เลย… (TH/EN)")

if query:
    add_message("user", query)
    with st.chat_message("user"):
        st.markdown(query)

    where = build_where_from_filters(selected_cats)
    rag_mode = "RAG" if mode.startswith("📚") else "LLM"

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                res = answer(query, mode=rag_mode, k=top_k, where=where)
            except Exception as e:
                res = {"answer": f"เกิดข้อผิดพลาด: {e}", "citations": [], "retrieval": []}

        st.markdown(res["answer"])

        # Citations
        if res.get("citations"):
            with st.expander("References"):
                for c in res["citations"]:
                    st.markdown(f"- {c}")

        # Retriever debug table
        if show_debug and res.get("retrieval"):
            import pandas as pd
            rows = []
            for r in res["retrieval"]:
                m = r["meta"]
                rows.append({
                    "title": m.get("title", ""),
                    "category": m.get("category", ""),
                    "section": m.get("section", ""),
                    "chunk#": m.get("chunk_index", ""),
                    "distance": round(float(r.get("distance", 0.0)), 4),
                    "url": m.get("url", "")
                })
            df = pd.DataFrame(rows)
            st.markdown("##### Retriever debug")
            st.dataframe(df, use_container_width=True)

        add_message("assistant", res["answer"])
