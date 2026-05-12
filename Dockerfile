FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    APP_CONFIG=configs/app.yaml \
    MODEL_PATH=models/best.pt \
    PRELOAD_MODEL=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloudrun.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0+cpu torchvision==0.26.0+cpu \
    && pip install --no-cache-dir -r requirements-cloudrun.txt

COPY . .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT', '8080'), timeout=3).read()"

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
