# scripts/try_rag.py
import sys, os
from pathlib import Path
from dotenv import load_dotenv

# ให้ Python มองเห็นแพ็กเกจ app/
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# โหลด .env ก่อนอ่านค่าตัวแปร
load_dotenv()

print("CWD               :", Path.cwd())
print("ENV CHROMA        :", os.getenv("CHROMA_PERSIST_DIR"))
print("ENV COLLECTION    :", os.getenv("COLLECTION_NAME"))

from app.rag.retriever import get_collection
col = get_collection()

try:
    cnt = col.count()
except Exception as e:
    cnt = f"error: {e}"
print("COLLECTION COUNT  :", cnt)

# ทดสอบ RAG
from dotenv import load_dotenv
load_dotenv()

import os
from chromadb.config import Settings
import chromadb
from pathlib import Path
print("CWD               :", Path.cwd())
print("ENV CHROMA        :", os.getenv("CHROMA_PERSIST_DIR"))
print("ENV COLLECTION    :", os.getenv("COLLECTION_NAME"))

chroma_path = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=True))

cols = client.list_collections()
print("FOUND COLLECTIONS :", [c.name for c in cols])
for c in cols:
    try:
        print(" -", c.name, "count=", c.count())
    except Exception as e:
        print(" -", c.name, "count error:", e)

from app.rag.pipeline import answer

if __name__ == "__main__":
    q = "แนวโน้ม AI สำหรับธุรกิจสมัยใหม่"
    res = answer(q, mode="RAG", k=5)
    print("\n=== ANSWER ===\n", res["answer"])
    print("\n=== REFERENCES ===")
    for c in res["citations"]:
        print("-", c)
