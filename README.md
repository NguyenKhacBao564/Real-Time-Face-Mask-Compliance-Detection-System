# Real-Time Face Mask Compliance Detection System

Project portfolio AI Engineer: train and deploy a real-time face mask compliance detector.

## Live Cloud Run demo

- Web demo: <https://facemask-compliance-szbjef7jsa-as.a.run.app>
- Health check: <https://facemask-compliance-szbjef7jsa-as.a.run.app/health>
- Event review API: <https://facemask-compliance-szbjef7jsa-as.a.run.app/api/v1/events?limit=10>
- REST image inference: `POST https://facemask-compliance-szbjef7jsa-as.a.run.app/api/v1/predict/image`
- WebSocket webcam inference: `wss://facemask-compliance-szbjef7jsa-as.a.run.app/api/v1/ws/detect`

Latest Cloud Run redeploy smoke test: 2026-06-08. `/health`, REST image
inference, WebSocket inference, `/api/v1/events`, and
`/api/v1/events/{event_id}` were verified after deploy. The deployed service is
configured with `min-instances=0` to avoid idle compute cost, so the first
request after inactivity may include cold-start latency.

The target is not a notebook-only demo. The project should prove four things:

1. Train or fine-tune an object detection model.
2. Prepare and audit a YOLO-format dataset.
3. Run image and webcam inference in real time.
4. Deploy a Dockerized API with measurable FPS, latency, and detection metrics.

## Problem

Detect face mask compliance with three classes:

```text
correct_mask
incorrect_mask
no_mask
```

The model should detect the face region and classify the mask status in one step. The MVP uses a single-stage YOLO detector, not a separate face detector plus mask classifier.

## MVP Scope

Build only the parts that matter for a strong, finishable portfolio project:

```text
Required:
- YOLO 3-class detector
- REST image inference endpoint
- WebSocket webcam inference endpoint
- simple browser frontend with canvas bounding boxes
- Docker deployment
- FPS, latency, mAP, precision, recall reporting

Optional after MVP:
- video upload
- YOLOv8s / YOLO11 comparison
- ONNXRuntime optimization
- public demo server

Out of scope:
- face recognition
- identity tracking
- multi-camera production system
- database-heavy dashboard
- email/SMS alerting
```

## Architecture

```text
Browser webcam or image upload
        |
        v
FastAPI REST / WebSocket API
        |
        v
YOLO detector
        |
        v
detections JSON: class, confidence, bbox, counts, latency
        |
        v
frontend canvas draws boxes and status counts
```

For realtime WebSocket, the default response should be JSON detections, not a base64 annotated frame. Returning annotated images every frame is heavier and should be kept as a debug option only.

## Recommended Stack

```text
Python
PyTorch
Ultralytics YOLO
OpenCV
FastAPI
WebSocket
Docker / Docker Compose
Vast.ai for GPU training
```

Baseline model:

```text
YOLOv8n
```

Use `YOLOv8n` first because it is small, fast, easy to train, and realistic for CPU/GPU demos. Larger models are optional only after the full pipeline works.

## Dataset Plan

Dataset strategy is documented in [Dataset.md](Dataset.md).

Short version:

1. Use `andrewmvd/face-mask-detection` only as a smoke test if needed.
2. Use PWMFD as the first real 3-class baseline dataset.
3. Use FMLD later to improve `incorrect_mask` robustness.
4. Do not merge datasets blindly; audit class balance and label quality first.
5. Treat `MaskedFace-Net` as a future classification/pretraining resource, not the MVP YOLO detection dataset.

The main project risk is the `incorrect_mask` class. It is usually much rarer than `correct_mask` and `no_mask`, so results must report per-class precision and recall honestly.

## Vast.ai Training Plan

Training workflow is documented in [TRAINING_VASTAI.md](TRAINING_VASTAI.md).

Short version:

1. Keep dataset download and conversion reproducible with scripts.
2. On Vast.ai, download data into `data/raw/`.
3. Convert/audit into `data/mask_detection/`.
4. Train YOLO from `data/mask_detection/data.yaml`.
5. Save only metrics, config, and download links in Git; store large weights externally.

## Deployment

Deployment workflow is documented in [DEPLOYMENT.md](DEPLOYMENT.md).

Short version:

```bash
docker compose up --build
```

Model weights are mounted from `models/` at runtime. They are not committed to Git and are not baked into the Docker image.

Cloud Run deployment helper:

```bash
PROJECT_ID=<your-gcp-project-id> REGION=asia-southeast1 MIN_INSTANCES=0 \
  ./scripts/deploy_cloudrun.sh
```

For a demo that should avoid cold starts, set `MIN_INSTANCES=1`. For portfolio
links that may sit idle, prefer `MIN_INSTANCES=0` and mention possible cold
start latency.

## Real-Time Target

Use explicit benchmark targets so "real-time" is measurable:

| Environment | Target |
| --- | --- |
| Local laptop CPU | at least 8-12 FPS at 640px input |
| Vast.ai / GPU demo | at least 20 FPS at 640px input |
| Cheap CPU server | at least 5 FPS with frame skipping |
| Latency report | average and p95 latency, with hardware specs |

If the project misses a target, keep the number and explain the bottleneck. Honest measurement is better than an inflated demo result.

## Target Repository Structure

```text
realtime-face-mask-detection/
├── configs/
│   ├── train.yaml
│   └── app.yaml
├── data/
│   └── README.md
├── models/
│   └── README.md
├── src/
│   ├── api/
│   ├── inference/
│   ├── training/
│   └── utils/
├── frontend/
├── scripts/
├── outputs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── Dataset.md
└── TRAINING_VASTAI.md
```

Recommended ignored paths:

```text
data/raw/
data/mask_detection/
runs/
outputs/
*.pt
*.onnx
models/*.pt
models/*.onnx
```

## Milestones

### Week 1: Dataset and Baseline

Deliverables:

```text
dataset audit report
YOLO-format data.yaml
YOLOv8n baseline training run
initial mAP / precision / recall
sample predictions
```

### Week 2: 3-Class Model and Local Inference

Deliverables:

```text
3-class fine-tuned model
per-class metrics
predict image script
local webcam script
local FPS and latency benchmark
```

### Week 3: API and Realtime Frontend

Deliverables:

```text
GET /health
POST /api/v1/predict/image
WS /api/v1/ws/detect
browser webcam client
canvas bounding boxes
snapshot logging for no_mask / incorrect_mask
```

### Week 4: Docker, Deployment, and Portfolio Polish

Deliverables:

```text
Dockerfile
docker-compose.yml
server benchmark
deployment notes
demo video or GIF
final README results table
CV-ready project bullet
```

## Metrics

Minimum report:

| Metric | Notes |
| --- | --- |
| mAP@50 | validation/test detection score |
| mAP@50:95 | stricter detection score |
| Precision | overall and per class |
| Recall | overall and per class |
| Local FPS | webcam or benchmark video |
| Server FPS | deployed environment |
| Latency | average and p95 if possible |
| Model size | `.pt` and exported model size |

Recommended result tables:

```text
Model comparison:
- smoke-test YOLOv8n on andrewmvd
- YOLOv8n on PWMFD
- optional YOLOv8s or ONNXRuntime benchmark

Per-class metrics:
- correct_mask precision / recall
- incorrect_mask precision / recall
- no_mask precision / recall
```

## Privacy And Ethics

This project detects mask compliance on face regions, but it must not identify people. The MVP should not store names, embeddings, or face recognition data. Violation snapshots are for debugging/demo only and should be disabled or cleaned before public deployment unless all subjects consent. Any demo video should use the owner, consenting friends, or synthetic/sample footage. Dataset licenses and attribution must be documented before publishing trained weights.

## Final CV Direction

Target bullet:

```text
Built and deployed a real-time face mask compliance detection system using YOLO, FastAPI WebSocket, OpenCV, and Docker, supporting live webcam inference with FPS/latency monitoring and violation snapshot logging.
```

## System & App Improvements for AI Camera Inference

The detector is wrapped in a small inference service that behaves more like a
real AI camera monitoring pipeline: every meaningful violation becomes a
structured **event** with a snapshot, exposed through a lightweight review API.
None of these changes touch the model — they sit on top of the existing
FastAPI + WebSocket inference paths.

### Architecture

```text
Browser webcam / image upload / curl
              |
              v
FastAPI (REST + WebSocket)
              |
              v
YOLOv8n MaskDetector  ──►  detections JSON (class, confidence, bbox, latency)
              |
              v
src/utils/events.py
  - filters: label ∈ {no_mask, incorrect_mask} AND confidence ≥ threshold
  - per-(client, label) cooldown
  - append-only JSONL log
  - JPEG snapshot of the violation frame
              |
              v
GET /api/v1/events             ── list / filter
GET /api/v1/events/{id}        ── detail
GET /api/v1/events/{id}/snapshot ── evidence image
```

### REST + WebSocket inference flow

* `POST /api/v1/predict/image?client_id=<optional>` — runs inference, then
  calls the event emitter. Response keeps the original shape and additively
  includes `frame_width`, `frame_height`, and `events` (a list of event ids,
  only when violations were emitted).
* `WS /api/v1/ws/detect?client_id=<optional>` — unchanged response shape. The
  optional `events` list and the legacy `snapshot_path` field are added only
  when a violation event was just persisted. If no `client_id` is supplied, the
  server auto-generates a 12-char `session_id` for logging.

### Event logging behavior

A detection is logged as an event when **all** of the following are true:

* class label is `no_mask` or `incorrect_mask`
* confidence ≥ `EVENT_CONFIDENCE_THRESHOLD` (default `0.6`)
* the per-`(client_id, label)` cooldown window has elapsed

Each event row (`outputs/events/events.jsonl`) contains:

```text
event_id, timestamp (UTC ISO-8601), source (image|websocket),
label, confidence, bbox, frame_width, frame_height,
latency_ms, snapshot_path (nullable), client_id, session_id
```

### Snapshot evidence behavior

* Snapshots are only saved when the event survives the threshold + cooldown
  filter, never for clean frames.
* Files land in `outputs/events/snapshots/` with the naming convention
  `YYYYMMDDThhmmssffffffZ_<label>_<event_id_prefix>.jpg`.
* `EVENT_SNAPSHOT_COOLDOWN_SECONDS` (default `3s`) caps how often a continuous
  violation can produce a new snapshot for the same `(client_id, label)` pair.
* Snapshots can be turned off without disabling event logging via
  `EVENT_SNAPSHOT_ENABLED=false`.

### Review API examples

```bash
# Recent events (newest first)
curl -s 'http://localhost:8000/api/v1/events?limit=10' | jq

# Only no_mask violations from a specific client/session
curl -s 'http://localhost:8000/api/v1/events?label=no_mask&client_id=demo' | jq

# Time-bounded query
curl -s 'http://localhost:8000/api/v1/events?start_time=2024-05-30T00:00:00Z' | jq

# Inspect one event
curl -s http://localhost:8000/api/v1/events/<event_id> | jq

# Download the evidence JPEG
curl -s http://localhost:8000/api/v1/events/<event_id>/snapshot -o evidence.jpg
```

### Latency / FPS logging

REST and WebSocket use Python `logging` (no Prometheus). Fields:

* REST per request: `inference_ms`, `request_ms`, `detections`,
  `violations_emitted`, `client_id`.
* WebSocket per session: `frames`, rolling `fps` every 50 frames, plus a final
  `ws_disconnect` log line with total frames, elapsed seconds, average FPS, and
  total violation events emitted. Each violation batch is also logged
  separately by the event module (`events_emitted ... count=N labels=...`).

### Configuration

All values read from environment variables (with defaults). See
[`.env.example`](.env.example) and the `events:` block in
[`configs/app.yaml`](configs/app.yaml):

| Variable | Default | Purpose |
| --- | --- | --- |
| `EVENT_LOG_ENABLED` | `true` | Disable to silence the whole event pipeline. |
| `EVENT_CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence to log an event. |
| `EVENT_SNAPSHOT_ENABLED` | `true` | Turn JPEG snapshots on/off. |
| `EVENT_SNAPSHOT_COOLDOWN_SECONDS` | `3` | Per-(client, label) cooldown. |
| `EVENT_OUTPUT_DIR` | `outputs/events` | Holds `events.jsonl` and `snapshots/`. |

### How to run locally

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
# open http://localhost:8000/
```

The Violations panel in the browser frontend now polls `/api/v1/events` every
3 seconds and shows each persisted event with a `snapshot` link, so the demo
UI matches what the API actually stores.

### How to test with curl

End-to-end recipes (health, image inference, event listing, snapshot
retrieval) live in [docs/testing_project2.md](docs/testing_project2.md). Unit
tests for the event module and the review API:

```bash
pytest tests/test_events.py -v
```

### Why this is closer to a production-style AI camera service

* Detections are turned into auditable **events**, not just transient WS
  messages — a real monitoring workflow needs durable evidence.
* Snapshots provide reviewable evidence per violation without persisting every
  frame (privacy- and disk-friendly).
* The review API is the contract a downstream dashboard, alerting worker, or
  human reviewer would actually consume.
* Cooldown + confidence threshold are the same primitives industrial AI camera
  systems use to keep alert volume sane.
* Storage stays JSONL on disk, with a single FastAPI worker — easy to migrate
  to SQLite/Postgres later without changing call sites.

## References

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [FastAPI WebSocket documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [Docker Compose documentation](https://docs.docker.com/compose/)
- [PWMFD dataset](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset)
- [FMLD dataset](https://github.com/borutb-fri/FMLD)
