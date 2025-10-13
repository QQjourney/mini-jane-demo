# app/rag/pipeline.py
from __future__ import annotations
from typing import Dict, List, Optional
import os

from dotenv import load_dotenv
import google.generativeai as genai

from .retriever import search
from .prompts import build_prompt, make_citations

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ตั้ง default ให้เป็นรุ่นใหม่ที่รองรับ generateContent แน่ ๆ
REQUESTED_MODEL = os.getenv("GENERATION_MODEL", "models/gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

genai.configure(api_key=GEMINI_API_KEY)


def _resolve_model_name(requested: str) -> str:
    """
    ทำให้ชื่อโมเดล robust:
    - รับทั้งรูปแบบสั้น ("gemini-2.5-flash") และเต็ม ("models/gemini-2.5-flash")
    - ถ้าตั้งไม่ตรง ให้ fallback เป็นรุ่นยอดนิยมที่ key ใช้งานได้
    """
    # ตัวเลือกตามลำดับความน่าจะใช้ได้
    candidates = []
    if requested:
        if requested.startswith("models/"):
            candidates.append(requested)
        else:
            candidates.append("models/" + requested)
            if not requested.endswith("-latest"):
                candidates.append("models/" + requested + "-latest")

    # fallback ปลอดภัย
    candidates += [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
    ]

    # ดึงรายการโมเดลที่รองรับ generateContent สำหรับ key นี้
    available = {
        m.name for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    }

    for name in candidates:
        if name in available:
            return name

    # ถ้าไม่เจอเลย ให้บอกชื่อที่ใช้ได้กลับไป
    raise RuntimeError(
        "No compatible Gemini model found for your API key.\n"
        "Please set GENERATION_MODEL in .env to one of:\n- " + "\n- ".join(sorted(available))
    )


# สร้างโมเดล 1 ครั้ง ใช้ซ้ำ (ลดเวลาและ error)
_MODEL_NAME = _resolve_model_name(REQUESTED_MODEL)
# หมายเหตุ: ใส่ system_instruction ตอนสร้างโมเดลภายหลัง (ในฟังก์ชัน answer)
# เพราะเราต้องรู้ภาษาจาก query ก่อน
print(f"[RAG] Using Gemini model: {_MODEL_NAME}")


def _call_llm(system_inst: str, user_text: str) -> str:
    """
    เรียก Gemini โดยกำหนด system prompt ผ่าน system_instruction
    """
    model = genai.GenerativeModel(model_name=_MODEL_NAME, system_instruction=system_inst)
    resp = model.generate_content(user_text)
    return resp.text if hasattr(resp, "text") else str(resp)


def answer(
    query: str,
    mode: str = "RAG",          # "RAG" | "LLM"
    k: int = 5,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
) -> Dict:
    """
    Return: { "answer": str, "citations": [..], "used_k": int, "retrieval": [...] }
    """
    if mode not in ("RAG", "LLM"):
        mode = "RAG"

    if mode == "LLM":
        # LLM ปกติ (ไม่มีบริบท RAG)
        text = _call_llm(system_inst="You are a helpful bilingual assistant.", user_text=query)
        return {"answer": text, "citations": [], "used_k": 0, "retrieval": []}

    # -------- RAG mode --------
    results = search(
        query, k=k, where=where, where_document=where_document,
        use_mmr=True, fetch_k=max(25, k * 4)
    )

    if not results:
        return {
            "answer": "ไม่พบข้อมูลในคลังความรู้สำหรับคำถามนี้ครับ/ค่ะ (No relevant information found).",
            "citations": [], "used_k": 0, "retrieval": []
        }

    prompt = build_prompt(query, results, lang="auto")
    # ส่ง system เป็น system_instruction และ user เป็นข้อความเดียว
    text = _call_llm(system_inst=prompt["system"], user_text=prompt["user"])
    cites = make_citations(results).splitlines()

    return {
        "answer": text,
        "citations": cites,
        "used_k": len(results),
        "retrieval": results
    }
