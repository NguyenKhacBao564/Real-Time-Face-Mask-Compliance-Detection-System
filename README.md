# Real-Time Face Mask Compliance Detection System

Project portfolio AI Engineer: train and deploy a real-time face mask compliance detector.

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

1. Use a YOLO-format dataset for the first baseline.
2. Use a 3-class dataset to add or fine-tune `incorrect_mask`.
3. Do not merge datasets blindly.
4. Do not commit raw datasets, model weights, training runs, or generated outputs to Git.
5. Treat `MaskedFace-Net` as a future classification/pretraining resource, not the MVP YOLO detection dataset.

## Vast.ai Training Plan

Training workflow is documented in [TRAINING_VASTAI.md](TRAINING_VASTAI.md).

Short version:

1. Keep dataset download and conversion reproducible with scripts.
2. On Vast.ai, download data into `data/raw/`.
3. Convert/audit into `data/mask_detection/`.
4. Train YOLO from `data/mask_detection/data.yaml`.
5. Save only metrics, config, and download links in Git; store large weights externally.

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

## Final CV Direction

Target bullet:

```text
Built and deployed a real-time face mask compliance detection system using YOLO, FastAPI WebSocket, OpenCV, and Docker, supporting live webcam inference with FPS/latency monitoring and violation snapshot logging.
```

## References

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [FastAPI WebSocket documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [Docker Compose documentation](https://docs.docker.com/compose/)
