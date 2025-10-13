# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps (lxml, trafilatura บางเวอร์ชันต้องมี)
RUN apt-get update && apt-get install -y \
    build-essential curl git libxml2-dev libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*

# requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# project
COPY app /app/app
COPY scripts /app/scripts
COPY notebooks /app/notebooks
COPY tests /app/tests
COPY README.md /app/README.md

# เลือกโหมดรันด้วยตัวแปร START_TARGET: api | ui
ENV START_TARGET=api
ENV PORT_API=8000
ENV PORT_UI=8501

# healthcheck ง่ายๆ
HEALTHCHECK CMD curl -f http://localhost:${PORT_API}/health || exit 1

# entry script
COPY <<'BASH' /app/start.sh
#!/usr/bin/env bash
set -e
echo "[start] mode=$START_TARGET"
if [ "$START_TARGET" = "ui" ]; then
  exec streamlit run app/ui/streamlit_app.py --server.port ${PORT_UI} --server.address 0.0.0.0
else
  exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT_API}
fi
BASH
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
