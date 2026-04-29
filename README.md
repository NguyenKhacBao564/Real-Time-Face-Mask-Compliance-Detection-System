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

## References

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [FastAPI WebSocket documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [Docker Compose documentation](https://docs.docker.com/compose/)
- [PWMFD dataset](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset)
- [FMLD dataset](https://github.com/borutb-fri/FMLD)
