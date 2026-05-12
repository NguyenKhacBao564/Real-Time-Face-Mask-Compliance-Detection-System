# Deployment Guide

This app can be deployed as a Dockerized FastAPI service. The Docker image should contain code and dependencies only. Model weights, datasets, outputs, and training runs stay outside the image.

## Important Browser Note

Webcam access requires a secure browser context:

```text
localhost works over HTTP
remote public server needs HTTPS
```

If you open `http://<server-ip>:8000` from your laptop, the browser may block camera access. For public webcam demo, use HTTPS through a domain/reverse proxy, or run the demo locally.

## Local Docker Test

Make sure model files exist:

```bash
ls -lh models/best.pt models/best.onnx
```

Build and run:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

## VPS / EC2 / Vast.ai Temporary Deployment

On the server:

```bash
git clone <your-repo-url>
cd Real-Time-Face-Mask-Compliance-Detection-System
mkdir -p models outputs
```

Copy model weights to the server:

```bash
scp models/best.pt root@<host>:/workspace/Real-Time-Face-Mask-Compliance-Detection-System/models/best.pt
scp models/best.onnx root@<host>:/workspace/Real-Time-Face-Mask-Compliance-Detection-System/models/best.onnx
```

If SSH uses a custom port:

```bash
scp -P <port> models/best.pt root@<host>:/workspace/Real-Time-Face-Mask-Compliance-Detection-System/models/best.pt
scp -P <port> models/best.onnx root@<host>:/workspace/Real-Time-Face-Mask-Compliance-Detection-System/models/best.onnx
```

Run:

```bash
docker compose up --build -d
docker compose logs -f api
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## What Not To Put In Docker Image

Do not bake these into the Docker image:

```text
data/raw/
data/mask_detection/
runs/
outputs/
models/best.pt
models/best.onnx
```

They are ignored by `.dockerignore` and should be mounted/copied at runtime.

## Recommended Public Demo Path

For portfolio, use one of these:

```text
Option A: Local demo video
- Run locally with webcam
- Record 60 seconds
- Put GIF/video in README

Option B: Public API demo
- Deploy FastAPI server
- Keep image upload endpoint available
- Use HTTPS if webcam frontend must work remotely
```

Local demo video is usually enough for an intern portfolio and avoids HTTPS/camera friction.
