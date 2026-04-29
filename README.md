Được. Ta sẽ lên kế hoạch theo hướng **project portfolio intern AI Engineer**, không phải chỉ “train YOLO xong”. Mục tiêu là có một project có thể ghi vào CV giống kiểu Hieu đã làm, nhưng tốt hơn ở phần **real-time + deploy + benchmark**.

# Project: Real-Time Face Mask Compliance Detection

## 1. Ý tưởng chính

Hệ thống nhận hình ảnh từ webcam/video/camera, phát hiện khuôn mặt/người và phân loại trạng thái khẩu trang:

```text
correct_mask      → đeo khẩu trang đúng
incorrect_mask    → đeo sai, hở mũi/miệng
no_mask           → không đeo khẩu trang
```

Pipeline:

```text
Webcam / video stream
        ↓
Frame capture
        ↓
YOLO detection
        ↓
Mask status classification
        ↓
Draw bounding boxes
        ↓
Count people by status
        ↓
Real-time dashboard
        ↓
Save violation snapshots
        ↓
Deploy server
```

Tên project nên dùng:

```text
Real-Time Face Mask Compliance Detection System
```

Hoặc chuyên nghiệp hơn:

```text
Real-Time PPE Compliance Monitoring System
```

Nhưng giai đoạn đầu nên giữ scope là **face mask** để làm nhanh và deploy chắc.

---

# 2. Mục tiêu của project

Project này phải chứng minh 5 năng lực:

```text
1. Biết train/fine-tune object detection model
2. Biết xử lý dataset YOLO format
3. Biết realtime inference bằng webcam/video
4. Biết deploy thành web app/server API
5. Biết đo FPS, latency, mAP, precision/recall
```

CV không nên chỉ ghi:

```text
Built face mask detection using YOLO.
```

Mà nên hướng tới bullet như:

```text
Built and deployed a real-time face mask compliance detection system using YOLO, FastAPI WebSocket, OpenCV, and Docker, supporting live webcam inference with FPS/latency monitoring and violation snapshot logging.
```

Đây mới là project có giá trị.

---

# 3. Scope MVP

## MVP bắt buộc

```text
Input:
- image upload
- video upload
- webcam real-time

Model:
- YOLOv8n hoặc YOLO11n
- 3 classes: correct_mask, incorrect_mask, no_mask

Backend:
- FastAPI
- REST endpoint cho image inference
- WebSocket endpoint cho realtime webcam

Frontend:
- Streamlit hoặc React đơn giản
- hiển thị video realtime + bounding box + count

Deployment:
- Docker
- deploy lên VPS/AWS EC2/Render/Railway/Vast demo server

Metrics:
- mAP@50
- precision/recall từng class
- FPS
- latency per frame
```

## Không làm ở MVP

```text
- face recognition
- nhận diện danh tính người
- multi-camera production system
- alert qua email/SMS
- database phức tạp
- training model quá lớn
```

Tránh face recognition vì dễ dính privacy và không cần cho CV intern.

---

# 4. Dataset

## Dataset nên chọn

Bạn nên tìm dataset có sẵn trên:

```text
Kaggle
Roboflow Universe
Hugging Face Datasets
```

Từ khóa tìm:

```text
face mask detection yolo dataset
mask wearing dataset correct incorrect no mask
face mask detection roboflow
mask detection with incorrect mask dataset
```

Yêu cầu dataset:

```text
Format: YOLO preferred
Classes:
  0: with_mask / correct_mask
  1: without_mask / no_mask
  2: mask_weared_incorrect / incorrect_mask
Số lượng tối thiểu: 2,000–5,000 ảnh
Có train/val/test split
License cho phép dùng portfolio/public demo
```

Nếu dataset chỉ có 2 class:

```text
with_mask
without_mask
```

thì vẫn dùng được, nhưng project sẽ kém hay hơn. Tốt nhất có 3 class vì “incorrect_mask” làm project thực tế hơn.

---

# 5. Cấu trúc dataset YOLO

Chuẩn YOLO:

```text
data/mask_detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

`data.yaml`:

```yaml
path: data/mask_detection
train: images/train
val: images/val
test: images/test

names:
  0: correct_mask
  1: incorrect_mask
  2: no_mask
```

Một dòng label YOLO:

```text
class_id x_center y_center width height
```

Ví dụ:

```text
0 0.512 0.431 0.124 0.188
```

---

# 6. Model nên dùng

Tôi khuyên dùng:

```text
YOLOv8n trước
```

Lý do:

```text
nhẹ
dễ train
dễ export ONNX
chạy CPU được ở mức demo
phù hợp realtime
ít rủi ro hơn model lớn
```

Sau đó thử thêm:

```text
YOLOv8s
YOLO11n
YOLO11s
```

Bảng thử nghiệm nên có:

| Model   | mAP@50 | FPS local | FPS server | Model size |
| ------- | -----: | --------: | ---------: | ---------: |
| YOLOv8n |    ... |       ... |        ... |        ... |
| YOLOv8s |    ... |       ... |        ... |        ... |
| YOLO11n |    ... |       ... |        ... |        ... |

Nếu chỉ có thời gian, train một model `YOLOv8n` là đủ.

---

# 7. Train bằng Vast.ai

## Vast.ai dùng để làm gì?

Dùng Vast.ai để:

```text
train model
evaluate model
export ONNX
download weights
```

Không nên phụ thuộc Vast.ai để deploy lâu dài vì instance có thể không ổn định như server chính thức. Nhưng có thể dùng Vast.ai để demo deployment tạm thời nếu cần GPU.

---

## Chọn instance Vast.ai

Ưu tiên:

```text
GPU: RTX 3060 / 3070 / 3080 / 3090 / A4000
VRAM: >= 8GB
Disk: >= 30GB
CUDA image: PyTorch + CUDA
Network: ổn định
Price: rẻ nhưng không chọn máy quá unreliable
```

Với YOLOv8n/s, 8GB VRAM là đủ.

---

## Môi trường train

Dùng Docker image có PyTorch CUDA. Sau khi vào instance:

```bash
nvidia-smi
python --version
```

Cài dependencies:

```bash
pip install ultralytics opencv-python matplotlib pandas tqdm roboflow
```

Clone repo:

```bash
git clone https://github.com/<your-username>/realtime-face-mask-detection.git
cd realtime-face-mask-detection
```

Hoặc nếu chưa có repo thì tạo local trước, sau đó push.

---

## Lệnh train YOLO

Ví dụ:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/mask_detection/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=mask_yolov8n
```

Nếu GPU yếu:

```bash
batch=8
imgsz=512
```

Nếu dataset nhỏ:

```bash
epochs=50-80
patience=15
```

Nếu dataset lớn:

```bash
epochs=80-100
```

---

## Evaluate

```bash
yolo detect val \
  model=runs/train/mask_yolov8n/weights/best.pt \
  data=data/mask_detection/data.yaml \
  imgsz=640 \
  device=0
```

Cần lưu:

```text
mAP@50
mAP@50:95
precision
recall
confusion matrix
PR curve
```

---

## Export model

Export ONNX:

```bash
yolo export \
  model=runs/train/mask_yolov8n/weights/best.pt \
  format=onnx \
  imgsz=640
```

Sau đó lưu:

```text
models/best.pt
models/best.onnx
```

Trong GitHub không nên push file model quá lớn. Có thể:

```text
models/README.md ghi link download
hoặc dùng GitHub Release
hoặc Google Drive/Hugging Face model repo
```

---

# 8. Kiến trúc hệ thống deploy

## Option A — Dễ nhất: Streamlit app

```text
Browser
  ↓
Streamlit webcam/upload
  ↓
YOLO inference
  ↓
Display result
```

Ưu:

```text
làm nhanh
demo đẹp
dễ deploy
```

Nhược:

```text
không thể hiện backend engineering tốt bằng FastAPI
realtime webcam đôi khi không mượt
```

## Option B — Tốt cho CV: FastAPI + WebSocket + frontend

```text
Browser webcam
    ↓ frame JPEG/base64
FastAPI WebSocket
    ↓
YOLO/ONNX inference
    ↓
Detection JSON + annotated frame
    ↓
Browser display
```

Ưu:

```text
đúng chất AI Engineer hơn
có API
có realtime
có thể Docker hóa tốt
```

Nhược:

```text
làm lâu hơn Streamlit
```

Tôi khuyên chọn **Option B**.

---

# 9. API design

## REST image inference

```text
POST /api/v1/predict/image
```

Input:

```text
image file
```

Output:

```json
{
  "detections": [
    {
      "class_name": "no_mask",
      "confidence": 0.91,
      "bbox": [120, 80, 240, 220]
    }
  ],
  "counts": {
    "correct_mask": 3,
    "incorrect_mask": 1,
    "no_mask": 2
  },
  "latency_ms": 42.7
}
```

## WebSocket realtime

```text
WS /api/v1/ws/detect
```

Client gửi frame, server trả:

```json
{
  "frame_id": 102,
  "detections": [...],
  "counts": {...},
  "fps": 12.4,
  "latency_ms": 81.2,
  "annotated_frame": "base64..."
}
```

---

# 10. Repo structure

Nên tạo repo như sau:

```text
realtime-face-mask-detection/
├── configs/
│   ├── train.yaml
│   └── app.yaml
│
├── data/
│   ├── README.md
│   └── mask_detection/
│
├── models/
│   ├── README.md
│   ├── best.pt
│   └── best.onnx
│
├── src/
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   ├── postprocess.py
│   │   └── benchmark.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── websocket.py
│   │
│   ├── utils/
│   │   ├── visualization.py
│   │   ├── logger.py
│   │   └── config.py
│   │
│   └── app.py
│
├── frontend/
│   └── simple_webcam_client/
│
├── outputs/
│   ├── predictions/
│   ├── snapshots/
│   └── benchmarks/
│
├── scripts/
│   ├── download_dataset.py
│   ├── train_vast.sh
│   ├── export_onnx.sh
│   └── run_server.sh
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

Nếu muốn làm nhanh hơn, chưa cần React frontend. Có thể dùng HTML/JS đơn giản trong `frontend/`.

---

# 11. Deployment plan

## Local first

Chạy local:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Test Swagger:

```text
http://localhost:8000/docs
```

Test webcam client:

```text
http://localhost:8000
```

---

## Docker

`Dockerfile` tối thiểu:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:

```bash
docker build -t mask-detection-api .
```

Run:

```bash
docker run -p 8000:8000 mask-detection-api
```

---

## Deploy server

Nên deploy theo 2 hướng:

### Hướng 1: CPU server rẻ

Dùng:

```text
AWS EC2 t3.medium/t3.large
VPS
Render
Railway
```

Model:

```text
YOLOv8n ONNX
imgsz 416 hoặc 512
frame skipping
```

Mục tiêu:

```text
5–10 FPS demo
```

### Hướng 2: GPU server demo

Dùng:

```text
Vast.ai instance
Docker
public port
```

Mục tiêu:

```text
15–30 FPS demo
```

Nhược điểm là không ổn định dài hạn. Nhưng để quay demo video thì ổn.

---

# 12. Realtime optimization

Để chạy realtime tốt, đừng gửi 30 FPS lên server.

Nên dùng:

```text
client gửi 5–10 FPS
resize frame về 640 hoặc 512
server inference
server trả kết quả
```

Tối ưu:

```text
1. YOLOv8n thay vì YOLOv8m/l
2. ONNXRuntime
3. frame skipping
4. confidence threshold 0.35–0.5
5. không encode ảnh quá lớn
6. chỉ save snapshot khi có no_mask/incorrect_mask
```

---

# 13. Milestones 4 tuần

## Tuần 1 — Dataset + Baseline training

Mục tiêu:

```text
dataset YOLO format
train baseline YOLOv8n trên Vast.ai
có best.pt
có metrics ban đầu
```

Tasks:

```text
1. Chọn dataset
2. Chuẩn hóa class names
3. Kiểm tra labels
4. Train YOLOv8n 50 epochs
5. Evaluate
6. Export ONNX
```

Deliverable:

```text
best.pt
best.onnx
confusion matrix
mAP/precision/recall
sample predictions
```

---

## Tuần 2 — Inference app local

Mục tiêu:

```text
local inference chạy ổn với image/video/webcam
```

Tasks:

```text
1. Viết Predictor class
2. Viết image inference
3. Viết video inference
4. Viết webcam demo bằng OpenCV
5. Benchmark FPS local
```

Deliverable:

```text
python scripts/run_webcam.py
python scripts/benchmark.py
outputs/predictions/
```

---

## Tuần 3 — FastAPI realtime

Mục tiêu:

```text
có backend API + realtime websocket
```

Tasks:

```text
1. FastAPI /predict/image
2. FastAPI /predict/video nếu cần
3. WebSocket /ws/detect
4. Simple HTML webcam client
5. Snapshot logging
6. Swagger docs
```

Deliverable:

```text
uvicorn server chạy được
webcam realtime trong browser
snapshot no_mask lưu vào outputs/snapshots
```

---

## Tuần 4 — Docker + Deployment + README

Mục tiêu:

```text
deploy server và hoàn thiện portfolio
```

Tasks:

```text
1. Dockerfile
2. docker-compose.yml
3. Deploy lên VPS/Vast/AWS
4. Quay demo video
5. Viết README
6. Viết CV bullets
7. Benchmark server FPS/latency
```

Deliverable:

```text
GitHub repo sạch
demo video/GIF
server deployment notes
metrics table
CV bullet ready
```

---

# 14. README cần có gì?

README nên có cấu trúc:

```text
1. Project overview
2. Demo GIF/video
3. Architecture diagram
4. Dataset
5. Training on Vast.ai
6. Results
7. Real-time deployment
8. API documentation
9. How to run locally
10. Docker deployment
11. Limitations
12. Future work
```

Bảng result:

| Metric      | Value |
| ----------- | ----: |
| mAP@50      |   ... |
| mAP@50:95   |   ... |
| Precision   |   ... |
| Recall      |   ... |
| Local FPS   |   ... |
| Server FPS  |   ... |
| Avg latency |   ... |

---

# 15. Điểm cần tránh

Không để project thành notebook demo.

Tránh:

```text
train.ipynb duy nhất
không README
không Docker
không benchmark
không demo video
không API
không giải thích dataset
```

Project tốt phải chạy được bằng lệnh:

```bash
docker compose up
```

hoặc:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

# 16. Công nghệ ghi vào CV

Sau khi hoàn thiện, mục skill/project có thể ghi:

```text
Python, PyTorch, Ultralytics YOLO, OpenCV, FastAPI, WebSocket, Docker, ONNXRuntime, Vast.ai, Linux, Git
```

Nếu deploy AWS/VPS:

```text
AWS EC2 / Linux VPS, Nginx, Docker Compose
```

---

# 17. Kế hoạch chốt

Tôi đề xuất chốt project như sau:

```text
Project name:
Real-Time Face Mask Compliance Detection System

Dataset:
YOLO-format public dataset with 3 classes:
correct_mask, incorrect_mask, no_mask

Training:
Vast.ai GPU instance
YOLOv8n baseline
YOLOv8s optional
Export ONNX

App:
FastAPI backend
WebSocket realtime webcam inference
Simple browser frontend
Snapshot logging

Deployment:
Dockerized server
Deploy demo on Vast.ai GPU or VPS/AWS EC2

Metrics:
mAP@50, precision, recall, FPS, latency
```

Đây là project đủ đơn giản để hoàn thành nhanh, nhưng đủ “engineering” để nhà tuyển dụng thấy bạn biết deploy AI model thật.
