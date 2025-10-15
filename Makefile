# ---------- SIMPLE MAKEFILE ----------
# ใช้ได้ทั้ง Windows (Git Bash) และ Linux/Mac

.PHONY: install reindex api ui docker-up docker-down clean

install:
	python -m venv .venv
	.venv/Scripts/pip install --upgrade pip
	.venv/Scripts/pip install -r requirements.txt

reindex:
	.venv/Scripts/python -m scripts.reindex_from_chunks

api:
	.venv/Scripts/python -m uvicorn app.api.main:app --reload --port 8000

ui:
	.venv/Scripts/streamlit run app/ui/streamlit_app.py

