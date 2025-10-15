

````markdown
# 🤖 mini-jane-demo — RAG + Gemini (Bilingual TH/EN)

ระบบสาธิต **Retrieval-Augmented Generation (RAG)** สำหรับถาม-ตอบ/สรุป/สร้างบทความจากคลังบทความ “Jenosize Ideas”  
รองรับ **ไทย/อังกฤษ** แบบ end-to-end: ingest → clean → chunk → embed → index → retrieve → generate

- **UI:** Streamlit (chat-style)
- **API:** FastAPI (Render)
- **Vector DB:** ChromaDB (persisted)
- **LLM:** Google Gemini 2.5 Flash
- **Embedding:** Google `text-embedding-004` (768d)

**Live:**
- Streamlit UI: https://mini-jane-demo-021512025.streamlit.app/
- Render API: https://mini-jane-demo.onrender.com

---

## 0) Directory Structure (โครงสร้างสำคัญ)

```bash
mini-jane-demo/
│
├─ app/
│  ├─ api/
│  │  ├─ main.py            # FastAPI endpoints (/health, /ask, /search, /categories)
│  │  └─ schemas.py         # Pydantic models (request/response)
│  ├─ rag/
│  │  ├─ embeddings.py      # ฟังก์ชันฝังเวกเตอร์ (Gemini text-embedding-004)
│  │  ├─ retriever.py       # เชื่อม Chroma, ค้นเอกสาร (MMR, filters)
│  │  ├─ pipeline.py        # RAG pipeline (build prompt + call LLM)
│  │  ├─ prompts.py         # สร้าง system/user prompt (auto TH/EN)
│  │  └─ utils.py           # helper: clean/normalize ฯลฯ
│  └─ ui/
│     └─ streamlit_app.py   # Chat UI (bilingual, category filter, citations)
│
├─ notebooks/
│  ├─ 01_scrape.ipynb       #  เก็บบทความจากเว็บ
│  ├─ 02_clean.ipynb        #  clean/normalize/sort
│  ├─ 03_build_vectorstore.ipynb  #  chunk → embed → index (Chroma)
│  ├─ 04_eval_sanity.ipynb  #  ตรวจคุณภาพ retrieval แบบเร็ว ๆ
│  └─ vectorstore/          # ✅ ChromaDB persisted (ใช้บน Streamlit/Render)
│
├─ data/
│  ├─ raw/                  # ข้อมูลดิบจาก 01 (jsonl)
│  │  ├─ jenosize_articles.jsonl
│  │  └─ jenosize_articles_sorted.jsonl
│  └─ processed/            # ข้อมูลหลัง 02/03
│     ├─ articles_clean.jsonl
│     └─ chunks.jsonl       # ✅ input สำหรับสร้าง index (fallback reindex)
│
├─ requirements.txt
├─ .env.example
├─ .python-version          # ใช้ 3.12.x (เพื่อความเข้ากันได้ของ Chroma)
└─ README.md
````

> หมายเหตุ: ในโปรดักชันเราเก็บ `vectorstore/` ไว้ใน repo เพื่อให้ Render ใช้งานได้ทันที (ไม่พึ่ง LFS, ไม่ต้องโหลดโมเดลหนัก)

---

## 1) Data Flow (01 → 04) และตำแหน่งไฟล์

### 🟧 01_scrape.ipynb — Scrape

* **อินพุต**: ลิสต์ URL ของ 6 หมวด (Futurist, Understand People & Consumer, Transformation & Technology, Utility for Our World, Real-time Marketing, Experience the New World)
* **เทคนิค**: `requests` + retry, `trafilatura` ดึง main text/title, map หมวดจาก path ของ URL, กัน rate-limit
* **กติกาคุณภาพ**: ทิ้งหน้าที่เนื้อหาน้อย (< ~200 คำ)
* **ผลลัพธ์**:

  * `data/raw/jenosize_articles.jsonl` (1 บรรทัด = 1 บทความ)
  * `data/raw/jenosize_articles_sorted.jsonl` (จัดเรียง/เวอร์ชันคงที่)

> *ตัวอย่าง record*:

```json
{"url":"...","title":"...","text":"...","category":"Futurist","section":"futurist","language":"en"}
```

### 🟧 02_clean.ipynb — Clean & Normalize

* **อินพุต**: `data/raw/jenosize_articles_sorted.jsonl`
* **เทคนิค**: ตัดท้าย “Loading…”, normalize whitespace, กรองบทความสั้นมาก (เช่น < 120 คำ), เติมฟิลด์ที่จำเป็น
* **ผลลัพธ์**:

  * `data/processed/articles_clean.jsonl`  (บทความสะอาด)
  * (ตรวจสอบรายชื่อ/สถิติโดยรวม)

### 🟧 03_build_vectorstore.ipynb — Chunk + Embed + Index

* **อินพุต**: `articles_clean.jsonl`
* **เทคนิค**:

  * แบ่งเป็น chunk (ตามย่อหน้า/ความยาวเหมาะสม) พร้อม `chunk_index`
  * **Embedding**: (เดิม) เคยใช้ local `BAAI/bge-m3` → **เปลี่ยนเป็น** `Gemini text-embedding-004` (768d) เพื่อให้รันบน Render/Streamlit ได้ฟรีและเบา
  * สร้าง Chroma persistent collection: `jenosize-ideas`
* **ผลลัพธ์**:

  * `data/processed/chunks.jsonl` (สำรองไว้ reindex ได้)
  * `notebooks/vectorstore/`  ✅ (ตัวฐาน Chroma ใช้งานจริง)

> *ตัวอย่าง chunk*:

```json
{"url":"...","title":"...","category":"Futurist","section":"futurist","chunk_index":2,"language":"en","text":"..."}
```

### 🟧 04_eval_sanity.ipynb — Retrieval sanity check

* ทดสอบ query ตัวอย่าง (TH/EN) → ดู top-K + หมวดที่คืนมา
* แบบ quick & dirty เพื่อยืนยันว่า index ถูกต้อง, bilingual OK, และ filter ตาม category ใช้ได้

---

## 2) ทำไมเลือก RAG + LLM (ไม่ fine-tune)

* **RAG** เหมาะกับคอลเลกชันบทความที่อัปเดตได้เรื่อย ๆ: เพิ่ม/ลบบทความ → reindex ได้ทันที โดยไม่ต้องเทรนโมเดล
* **ไม่ต้อง labeling/annotation**: ลดใช้เวลาและงบเทียบกับ fine-tuning
* **ลด hallucination**: อ้างอิงจากคลังความรู้จริง + แสดง citations ให้ตรวจสอบได้
* **ปรับสเกลง่าย**: ย้าย vectorstore/เพิ่ม shards/เปลี่ยน retriever ได้ โดยไม่แตะตัว LLM

> ถ้าโจทย์เปลี่ยนเป็น “สไตล์การเขียนเฉพาะแบรนด์” ที่ฝังอยู่ใน model หรือต้องตอบสนอง domain ที่แคบมาก → ค่อยพิจารณา fine-tune ต่อยอด

---

## 3) โมเดลที่ใช้

* **LLM (Generate):** `models/gemini-2.5-flash`

  * เร็ว, คุ้มราคา, รองรับ TH/EN ดี
* **Embedding (Vector):** `models/text-embedding-004` (768 มิติ)

  * เบา/เสถียรบนโฮสติ้งฟรี
  * ใช้ร่วมกับ Chroma ได้ดี

> หมายเหตุ: เริ่มต้นเราใช้โมเดล local (`SentenceTransformer: BAAI/bge-m3`) แต่ฟรีโฮสติ้ง (Render/Streamlit Cloud) ไม่เหมาะกับการโหลดโมเดลหนัก ๆ + wheel ของบาง lib ไม่รองรับ CPU/OS → จึงย้ายมาใช้ **Gemini embedding API** และเก็บ **vectorstore** ที่สร้างเสร็จแล้วลง repo เพื่อให้ deploy ได้ทันที

---

## 4) สถาปัตยกรรมโดยรวม (Flow)

```mermaid
flowchart LR
U[User] -- TH/EN query --> S(Streamlit UI)
S -- POST /ask --> A(FastAPI API)
A --> R[Retriever (ChromaDB)]
R --> A
A --> L[Gemini 2.5 Flash]
L --> A
A --> S
S --> U

subgraph Pipeline
R -. uses .-> V[(ChromaDB\nnotebooks/vectorstore)]
A -. embed .-> E[Gemini text-embedding-004]
end
```

---

## 5) ใช้งานแบบ Local

### 5.1 เตรียมโปรเจกต์

```bash
git clone https://github.com/QQjourney/mini-jane-demo.git
cd mini-jane-demo

# Python 3.12.x
python -m venv .venv
.venv\Scripts\activate     # Windows PowerShell
pip install -r requirements.txt
```

### 5.2 ตั้งค่า `.env`

```ini
# ====== API Keys ======
GEMINI_API_KEY=YOUR_KEY

# ====== Embedding Config ======
GEMINI_EMBED_MODEL=models/text-embedding-004
LANG_MODE=multilingual
CHROMA_PERSIST_DIR=./notebooks/vectorstore
COLLECTION_NAME=jenosize-ideas

# ====== App Mode ======
APP_ENV=development
LOG_LEVEL=INFO
AUTO_REINDEX=0

# ====== Metadata ======
PROJECT_NAME=mini-jane-demo
DATA_PATH=./data
RAW_PATH=./data/raw
PROCESSED_PATH=./data/processed
GENERATION_MODEL=models/gemini-2.5-flash
```

> ถ้าไม่มี `notebooks/vectorstore/` ให้ตั้ง `AUTO_REINDEX=1` และตรวจว่ามี `data/processed/chunks.jsonl` → ระบบจะสร้าง index ให้เองตอนเริ่ม

### 5.3 รัน UI (Streamlit)

```bash
streamlit run app/ui/streamlit_app.py
# เปิด http://localhost:8501
```

### 5.4 รัน API (FastAPI)

```bash
uvicorn app.api.main:app --reload --port 8000
# Swagger: http://localhost:8000/docs
```

---

## 6) ใช้งานผ่านลิงก์ (ไม่ต้องติดตั้ง)

* **UI:** [https://mini-jane-demo-021512025.streamlit.app/](https://mini-jane-demo-021512025.streamlit.app/)
* **API:** [https://mini-jane-demo.onrender.com](https://mini-jane-demo.onrender.com)

  * `/health` → ตรวจว่ายังขึ้นอยู่ไหม
  * `/docs` → interactive docs

---

## 7) วิธีใช้ API (ขอตัวอย่าง JSON เท่านั้น)

### `POST /ask`

**Request JSON**

```json
{
  "query": "AI trends for 2030",
  "mode": "RAG",
  "k": 5
}
```

**Response JSON (ตัวอย่าง)**

```json
{
  "answer": "AI จะเป็นโครงสร้างพื้นฐานของธุรกิจปี 2030...",
  "citations": [
    "AI trends for 2030 — Futurist (Jenosize Ideas)",
    "Transformation & Technology — Market overview"
  ],
  "used_k": 5,
  "retrieval": [
    {
      "text": "…ข้อความ chunk…",
      "distance": 0.173,
      "meta": {
        "url": "https://www.jenosize.com/en/ideas/futurist/…",
        "title": "AI Trends 2030",
        "category": "Futurist",
        "section": "futurist",
        "chunk_index": 2,
        "language": "en"
      }
    }
  ]
}
```

### `POST /search`

**Request JSON**

```json
{"query": "consumer behaviour in thailand", "k": 5}
```

**Response JSON**

```json
{
  "results": [
    { "text": "…", "distance": 0.22, "meta": {"category":"Understand People & Consumer", "title":"…"} }
  ]
}
```

### `GET /categories`

**Response JSON**

```json
{
  "categories": [
    "Futurist",
    "Understand People & Consumer",
    "Transformation & Technology",
    "Utility for Our World",
    "Real-time Marketing",
    "Experience the New World"
  ]
}
```

---

## 8) Troubleshooting

| อาการ                              | สาเหตุ                             | วิธีแก้                                                                            |
| ---------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------- |
| `/health` 200 แต่ถามไม่ได้         | vectorstore ไม่ถูกอ่าน             | เช็ก `CHROMA_PERSIST_DIR` และมีไฟล์ใน `notebooks/vectorstore/` จริง                |
| ตอบว่า “ไม่พบข้อมูล”               | คลังว่าง/collection count=0        | ตั้ง `AUTO_REINDEX=1` และมี `data/processed/chunks.jsonl`, หรือ push `vectorstore` |
| ล้มตอน start บน Render             | Python 3.13 + chroma rust bindings | ตั้ง `PYTHON_VERSION=3.12.5`, `UVICORN_WORKERS=1`, `WEB_CONCURRENCY=1`             |
| ไลบรารีโหลดโมเดลฝั่ง local ช้า/ล้ม | ใช้ ST/BGE ใหญ่ไปบนโฮสติ้งฟรี      | ย้ายมาใช้ `text-embedding-004` + เก็บ vectorstore ใน repo                          |
| หมวดหมู่เรียก filter ไม่ติด        | เมตาดาต้า chunk ไม่มี category     | ย้อนดู 02/03 เติม field ให้ครบก่อน embed/index                                     |

---

## 9) License

MIT License — ดูไฟล์ `LICENSE`

---

## 10) Notes / Design Decision

* เลือก **RAG + LLM** เพื่อความคล่องตัว: เพิ่มบทความใหม่ → reindex จบ ไม่ต้องเทรน/label
* ใช้ **Gemini 2.5 Flash** เป็นตัวตอบ: เร็ว/คุ้ม/รองรับไทยดี
* ใช้ **Gemini text-embedding-004**: ฝั่ง embed เบาและเสถียรกว่าโหลดโมเดลใหญ่ในฟรีโฮสติ้ง
* เก็บ **vectorstore** ลง repo: ลดความเสี่ยง runtime ไม่เจอฟイル/โหลดไม่ทัน

```

> ถ้าจะเพิ่มรายชื่อ URL ที่ใช้ใน 01 ให้เติมลงในหัวข้อ 1) ได้เลยครับ (ผมเว้นช่องไว้ตามที่ขอ)  
> อยากแปะสกรีนช็อต UI/Flow เพิ่มใน README ก็วางใต้หัวข้อ Overview ได้เลย 👍
::contentReference[oaicite:0]{index=0}
```
