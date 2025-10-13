# app/rag/prompts.py
from typing import List, Dict

SYSTEM_PROMPT_TH = """คุณคือผู้ช่วยชื่อ mini-jane-demo ที่เชี่ยวชาญด้านธุรกิจ การตลาด เทคโนโลยี และแนวโน้มอนาคต
- ตอบเป็นภาษาไทยเมื่อผู้ใช้ถามเป็นไทย มิฉะนั้นตอบเป็นอังกฤษ
- ใช้เฉพาะข้อมูลจาก Context ที่ให้มา ถ้าไม่พบข้อมูล ให้บอกอย่างสุภาพว่า "ไม่พบข้อมูลในคลัง" (อย่าเดา)
- ให้สรุป/อธิบายอย่างกระชับ มี bullet หรือลำดับข้อได้
- แนบแหล่งอ้างอิง (title + category + url + chunk#) ด้านล่างเสมอ
"""

SYSTEM_PROMPT_EN = """You are mini-jane-demo, a helpful assistant specialized in business, marketing, technology, and future trends.
- Reply in English unless the user speaks Thai; if the user speaks Thai, reply in Thai.
- Use only the provided Context. If the answer is not found, say "No relevant information found in the knowledge base." Do not hallucinate.
- Keep answers concise and structured (bullets or short paragraphs).
- Always include citations (title + category + url + chunk#) at the end.
"""

def make_context_block(results: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        m = r["meta"]
        title = m.get("title","(no title)")
        cat   = m.get("category","-")
        url   = m.get("url","-")
        idx   = m.get("chunk_index","-")
        lines.append(f"[{i}] {title} [{cat}] ({url}#chunk{idx})\n{r['text']}")
    return "\n\n---\n\n".join(lines)

def make_citations(results: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        m = r["meta"]
        title = m.get("title","(no title)")
        cat   = m.get("category","-")
        url   = m.get("url","-")
        idx   = m.get("chunk_index","-")
        lines.append(f"[{i}] {title} · {cat} · {url}#chunk{idx}")
    return "\n".join(lines)

def build_prompt(user_query: str, results: List[Dict], lang: str = "auto") -> Dict[str,str]:
    # ตรวจภาษาแบบง่าย: ถ้ามีอักษรไทย -> th
    if lang == "auto":
        lang = "th" if any("\u0E00" <= ch <= "\u0E7F" for ch in user_query) else "en"

    system = SYSTEM_PROMPT_TH if lang=="th" else SYSTEM_PROMPT_EN
    context = make_context_block(results)
    citations = make_citations(results)

    user_block = f"""User Query:
{user_query}

Context:
{context}

Instructions:
- Answer strictly from Context above.
- Keep it concise; use bullets when helpful.
- End with a "References" section listing the items used, exactly as below:

References:
{citations}
"""
    return {"system": system, "user": user_block, "lang": lang}
