

---

````markdown
# 🤖 mini-jane-demo (RAG + Gemini Embedding)

ระบบสาธิต **Retrieval-Augmented Generation (RAG)** ที่ใช้ **Gemini Generative AI**  
เชื่อมกับ **ChromaDB Vectorstore** เพื่อให้ผู้ใช้ถามคำถาม (TH/EN) จากฐานความรู้ภายในองค์กร  
พร้อมทั้งมีทั้ง **Frontend (Streamlit UI)** และ **Backend API (FastAPI on Render)**

---

## 📁 Overview & Directory Structure

```bash
mini-jane-demo/
│
├── app/
│   ├── api/                 # FastAPI backend (API endpoints)
│   │   ├── main.py          # จุดเริ่มต้นของ API
│   │   └── schemas.py       # Pydantic models สำหรับ request/response
│   │
│   ├── rag/                 # RAG pipeline
│   │   ├── embeddings.py    # โมดูลฝังเวกเตอร์ (Gemini text-embedding-004)
│   │   ├── retriever.py     # จัดการการค้นหาใน Chroma
│   │   ├── pipeline.py      # รวม logic ของการตอบคำถาม (RAG + LLM)
│   │   ├── prompts.py       # สร้าง prompt สำหรับ Gemini LLM
│   │   └── utils.py         # ตัวช่วยเล็ก ๆ เช่น normalize text
│   │
│   └── ui/
│       └── streamlit_app.py # Streamlit frontend (Chat interface)
│
├── data/
│   ├── raw/                 # ข้อมูลดิบ (ก่อนประมวลผล)
│   ├── processed/           # ข้อมูลหลัง chunk และ clean แล้ว
│   └── processed/chunks.jsonl
│
├── notebooks/
│   ├── vectorstore/         # ✅ ChromaDB persisted (ใช้ gemini embed)
│   └── 03_build_vectorstore.ipynb  # Notebook สำหรับสร้างฐานเวกเตอร์
│   
├── requirements.txt         # Python dependencies
├── .env.example             # ตัวอย่าง environment variables
├── .python-version          # ใช้ Python 3.12.0 (แก้ bug chromadb)
├── runtime.txt              # optional (ignored by Render)
├── README.md                # (ไฟล์นี้)
└── Dockerfile               # สำหรับ container deployment
````

---

## ⚙️ System Workflow

### 🔸 Flow Diagram

```
User Query → Streamlit UI → FastAPI /ask endpoint
     ↓                         ↓
  [Gemini text-embedding-004]  [Retriever: ChromaDB]
     ↓                         ↓
     └──> สร้าง embedding ----> ดึงเอกสารที่ใกล้เคียง
                                ↓
                          [Gemini 2.5 Flash]
                                ↓
                         สร้างคำตอบ (TH/EN)
                                ↓
                     ส่งกลับ UI พร้อม citations
```

---

## 🧠 System Description

1. **Chunking** — ข้อมูลใน `/data/processed/chunks.jsonl` ถูกแยกเป็นส่วน ๆ
2. **Embedding** — แต่ละ chunk ถูกฝังด้วย `Gemini text-embedding-004` → ได้เวกเตอร์ 768 มิติ
3. **Vectorstore** — เก็บใน `ChromaDB` แบบ Persistent (ใน `notebooks/vectorstore`)
4. **Retriever** — ใช้ cosine similarity + MMR ดึง top-K เอกสาร
5. **RAG Pipeline** — รวมเอกสาร + Prompt ให้ Gemini 2.5 Flash ตอบ
6. **Output** — ตอบกลับพร้อมอ้างอิง (citations)
7. **Frontend/UI** — Streamlit แสดงแชทและ debug mode
8. **API Layer** — FastAPI บน Render รองรับการเชื่อมระบบภายนอก

---

## 🚀 Installation & Usage

### 🧩 1. Clone & Run Locally

```bash
git clone https://github.com/QQjourney/mini-jane-demo.git
cd mini-jane-demo
python -m venv .venv
.venv\Scripts\activate   # (Windows)
pip install -r requirements.txt
```

### 🧠 2. สร้างไฟล์ `.env`

```bash
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_EMBED_MODEL=models/text-embedding-004
GENERATION_MODEL=models/gemini-2.5-flash
CHROMA_PERSIST_DIR=./notebooks/vectorstore
COLLECTION_NAME=jenosize-ideas
```

---

## 🖥️ 3. Run Streamlit UI

```bash
streamlit run app/ui/streamlit_app.py
```

เปิดที่ [http://localhost:8501](http://localhost:8501)

---

## 🌐 4. Deploy API (FastAPI + Render)

**Build Command**

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

**Start Command**

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
```

**Environment Variables บน Render**

```
PYTHON_VERSION=3.12.5
GEMINI_API_KEY=<KEY>
GENERATION_MODEL=models/gemini-2.5-flash
GEMINI_EMBED_MODEL=models/text-embedding-004
CHROMA_PERSIST_DIR=./notebooks/vectorstore
COLLECTION_NAME=jenosize-ideas
UVICORN_WORKERS=1
WEB_CONCURRENCY=1
```

### 🌍 URLs

* **Streamlit UI:** [https://mini-jane-demo-021512025.streamlit.app/](https://mini-jane-demo-021512025.streamlit.app/)
* **Render API:** [https://mini-jane-demo.onrender.com](https://mini-jane-demo.onrender.com)

---

## 🧩 Example API Usage (JSON Only)

### `POST /ask`

**Request:**

```json
{
  "query": "เทรนด์ AI ปี 2030",
  "mode": "RAG",
  "k": 5
}
```

**Response:**

```json
{
  "answer": "AI จะกลายเป็นส่วนหนึ่งของทุกอุตสาหกรรม...",
  "citations": [
    "AI Trends 2030 - Jenosize Report (2024)",
    "Transformation & Technology Section"
  ],
  "used_k": 5,
  "retrieval": [
    {"text": "...", "distance": 0.12, "meta": {"category": "Futurist"}}
  ]
}
```

---

### `POST /search`

**Request:**

```json
{"query": "Consumer behaviour", "k": 5}
```

**Response:**

```json
{
  "results": [
    {"text": "New trends in Gen Z...", "distance": 0.18, "meta": {"category": "Understand People & Consumer"}}
  ]
}
```

---

### `GET /categories`

**Response:**

```json
{
  "categories": [
    "Futurist",
    "Transformation & Technology",
    "Understand People & Consumer"
  ]
}
```

---

## 🧰 Troubleshooting

| ปัญหา                                      | สาเหตุ                                | วิธีแก้                                           |
| ------------------------------------------ | ------------------------------------- | ------------------------------------------------- |
| API 404                                    | เส้นทางผิด เช่น `/ask` แทน `/api/ask` | ตรวจ URL ที่ Render                               |
| `chromadb` error                           | Python 3.13 ไม่รองรับ                 | ตั้ง `PYTHON_VERSION=3.12.5`                      |
| “No relevant information found”            | ไม่มีฐานเวกเตอร์                      | ตรวจ `CHROMA_PERSIST_DIR` ว่าถูก push             |
| Streamlit ขึ้น error `SentenceTransformer` | ใช้ embed model ใหญ่เกิน              | เปลี่ยนเป็น `text-embedding-004`                  |
| Timeout ระหว่าง build                      | Render Starter Plan ช้า               | ใช้ `pip install --no-cache-dir` ใน build command |

---

## 📜 License

Distributed under the MIT License.
See [`LICENSE`](LICENSE) for more information.

---

## 🧩 Credits

Project by **Pattarit Sotyom**


* **Gemini 2.5 Flash + Text Embedding 004**
* **ChromaDB**
* **FastAPI**
* **Streamlit**
* **Render Cloud**

---

```

---

```
