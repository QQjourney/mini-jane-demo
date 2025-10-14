# syntax=docker/dockerfile:1
FROM python:3.12-slim

# ----- Env base -----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1200

# (ออปชัน) cache โหมด huggingface ให้ mount ได้จากภายนอก
# ENV HF_HOME=/app/.cache/hf

WORKDIR /app

# ----- System packages (สำหรับ lxml / trafilatura / build wheels) -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    libxml2-dev libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*

# ----- Python deps -----
# 1) ติดตั้ง PyTorch (CPU only) ก่อน เพื่อกันไปดึง CUDA ขนาดใหญ่
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu

# 2) ติดตั้ง deps อื่น ๆ จาก requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ----- Project code -----
COPY app /app/app
COPY scripts /app/scripts
COPY notebooks /app/notebooks
COPY tests /app/tests
COPY README.md /app/README.md
# (ไม่ copy .env / vectorstore / data เข้ามาใน image — ให้ mount ผ่าน volumes)

# พอร์ตที่มักใช้ (ออปชัน — compose จะ map เอง)
EXPOSE 8000 8501

# ไม่กำหนด CMD ที่นี่ เพราะจะสั่งผ่าน docker-compose:
# - API: ["python","-m","uvicorn","app.api.main:app","--host","0.0.0.0","--port","8000"]
# - UI : ["streamlit","run","app/ui/streamlit_app.py","--server.port","8501","--server.address","0.0.0.0"]
