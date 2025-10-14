# ===== Auto-reindex on first run (Streamlit Cloud friendly) =====
from pathlib import Path
import os, json, time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# --- ENV / PATH ---
CHROMA_DIR      = Path(os.getenv("CHROMA_PERSIST_DIR", "./vectorstore"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "jenosize-ideas")
CHUNKS_FILE     = Path(os.getenv("CHUNKS_FILE", "./data/processed/chunks.jsonl"))
EMBED_MODEL     = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# --- Cache resources: โหลดครั้งเดียวต่อโปรเซส ---
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL, device="cpu")  # Streamlit Cloud = CPU

@st.cache_resource(show_spinner=False)
def get_chroma():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(allow_reset=True))
    col = client.get_or_create_collection(name=COLLECTION_NAME)
    return client, col

def _make_id(url: str, chunk_index: int) -> str:
    # ให้ ID deterministic (กันซ้ำตอนรันใหม่)
    return f"{url}#chunk{chunk_index}"

def _load_chunks():
    assert CHUNKS_FILE.exists(), f"Not found: {CHUNKS_FILE}"
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # ขาด field ไหน ก็เติม default
            chunks.append({
                "url": rec.get("url",""),
                "title": rec.get("title",""),
                "category": rec.get("category",""),
                "section": rec.get("section",""),
                "chunk_index": int(rec.get("chunk_index", 0)),
                "text": rec.get("text",""),
                "language": rec.get("language","en"),
            })
    return chunks

def ensure_index_ready():
    """
    ถ้า collection ยังว่าง หรือ vectorstore ไม่มี ให้ reindex จาก chunks.jsonl อัตโนมัติ
    ใช้ progress bar ของ Streamlit เพื่อ feedback
    """
    client, col = get_chroma()

    try:
        n = col.count()
    except Exception:
        n = 0

    if n and n > 0:
        st.sidebar.info(f"🔎 Vector index ready: {n} chunks")
        return

    if not CHUNKS_FILE.exists():
        st.sidebar.error("❌ Missing chunks.jsonl – cannot build index.")
        return

    st.sidebar.warning("⚙️ Building vector index (first run). Please wait…")
    chunks = _load_chunks()

    # dedupe ตาม (url, chunk_index)
    seen = set()
    data = []
    for c in chunks:
        key = (c["url"], c["chunk_index"])
        if key in seen:  # กันซ้ำ
            continue
        seen.add(key)
        data.append(c)

    embedder = get_embedder()
    added = 0
    pbar = st.sidebar.progress(0, text="Indexing…")

    # batch insert
    BATCH = 64
    total = len(data)
    for i in range(0, total, BATCH):
        batch = data[i:i+BATCH]
        ids   = [_make_id(b["url"], b["chunk_index"]) for b in batch]
        docs  = [b["text"] for b in batch]
        metas = [{
            "url": b["url"], "title": b["title"], "category": b["category"],
            "section": b["section"], "chunk_index": b["chunk_index"], "language": b["language"]
        } for b in batch]

        embs = embedder.encode(docs, normalize_embeddings=True).tolist()
        # ใช้ upsert เพื่อ rerun ได้โดยไม่ error
        col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        added += len(batch)
        pbar.progress(min(1.0, added/total), text=f"Indexing… {added}/{total}")

    pbar.empty()
    st.sidebar.success(f"✅ Index built: {added} chunks")
    time.sleep(0.3)

# เรียกหนึ่งครั้งตอนโหลดแอป
ensure_index_ready()
# ===== End: Auto-reindex =====

# ------- Import our RAG pipeline -------
from app.rag.pipeline import answer
from app.rag.retriever import get_collection

st.set_page_config(page_title="mini-jane-demo", page_icon="🤖", layout="wide")

# -------- Utilities --------
def build_where_from_filters(selected_cats: List[str]) -> Optional[Dict]:
    if not selected_cats:
        return None
    if len(selected_cats) == 1:
        return {"category": selected_cats[0]}
    return {"category": {"$in": selected_cats}}

def ensure_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role":"user"/"assistant","content": "..."}]

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def get_all_categories(limit: int = 2000) -> List[str]:
    """Read some docs from Chroma to collect category values."""
    try:
        col = get_collection()
        # Prefer .get(limit=...) for portability
        res = col.get(limit=limit)
        metas = res.get("metadatas", []) or []
        cats = {m.get("category", "-") for m in metas if isinstance(m, dict)}
        cats.discard("-")
        items = sorted([c for c in cats if c])
        # fallback to known buckets if empty
        if not items:
            items = [
                "Futurist",
                "Understand People & Consumer",
                "Transformation & Technology",
                "Utility for Our World",
                "Real-time Marketing",
                "Experience the New World",
            ]
        return items
    except Exception:
        return [
            "Futurist",
            "Understand People & Consumer",
            "Transformation & Technology",
            "Utility for Our World",
            "Real-time Marketing",
            "Experience the New World",
        ]

# -------- Sidebar Controls --------
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio("Mode", options=["📚 RAG", "💬 LLM only"], index=0)
top_k = st.sidebar.slider("Top-K documents", min_value=3, max_value=10, value=5, step=1)

available_cats = get_all_categories()
selected_cats = st.sidebar.multiselect("Filter by category (optional)", options=available_cats, default=[])

show_debug = st.sidebar.checkbox("Show retriever debug", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Vector store: `{os.getenv('CHROMA_PERSIST_DIR', './vectorstore')}`")
st.sidebar.caption(f"Collection : `{os.getenv('COLLECTION_NAME', 'jenosize-ideas')}`")
st.sidebar.caption(f"Embed model: `{os.getenv('EMBED_MODEL', 'BAAI/bge-m3')}`")
st.sidebar.caption(f"Gen model  : `{os.getenv('GENERATION_MODEL', 'gemini-1.5-flash')}`")
st.sidebar.markdown("---")
st.sidebar.caption("Tip: พิมพ์ได้ทั้งไทย/อังกฤษ ระบบจะตอบตามภาษา")

# -------- Main Chat UI --------
st.title("mini-jane-demo 🤖")
st.caption("Chat with RAG over Jenosize Ideas (bilingual TH/EN) — with citations.")

ensure_session()

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("ถามคำถามได้เลย… (TH/EN)")

if query:
    add_message("user", query)
    with st.chat_message("user"):
        st.markdown(query)

    # Build filters
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

        # Retriever debug
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
