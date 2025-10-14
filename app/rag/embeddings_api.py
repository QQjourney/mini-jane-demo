# app/rag/embeddings_api.py
from __future__ import annotations
from typing import List, Literal
import os, math
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

genai.configure(api_key=GEMINI_API_KEY)

# ขนาด batch ที่ปลอดภัยบน Render/เน็ตทั่วไป (ปรับได้)
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

TaskType = Literal["retrieval_query", "retrieval_document"]

def _embed_one(text: str, task_type: TaskType) -> List[float]:
    # ตามสเปค: https://ai.google.dev/gemini-api/docs/embeddings
    resp = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=text,
        task_type=task_type,
    )
    return resp["embedding"]

def embed_texts(texts: List[str], task_type: TaskType) -> List[List[float]]:
    """
    ฝั่งไคลเอนต์ของ google.generativeai ยังไม่มี batch API ที่เสถียรเท่ากันทุกเวอร์ชัน
    จึง loop ทีละรายการ (หรือทีละชิ้นใน batch) พร้อมกันล้ม-ไปต่อ
    """
    out: List[List[float]] = []
    for t in texts:
        try:
            out.append(_embed_one(t, task_type))
        except Exception as e:
            # กันเหตุฉุกเฉิน: ถ้าพังให้เวกเตอร์ศูนย์แทน (หรือ raise ก็ได้)
            out.append([0.0] * 768)
    return out

def embed_query(text: str) -> List[float]:
    return embed_texts([text], "retrieval_query")[0]

def embed_queries(texts: List[str]) -> List[List[float]]:
    return embed_texts(texts, "retrieval_query")

def embed_docs(texts: List[str]) -> List[List[float]]:
    return embed_texts(texts, "retrieval_document")
