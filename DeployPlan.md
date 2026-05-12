# Deploy Plan: Google Cloud Run

## Mục tiêu

Deploy dự án `Real-Time-Face-Mask-Compliance-Detection-System` thành một demo public trên Google Cloud Run:

- FastAPI phục vụ API, WebSocket và frontend webcam trong cùng một service.
- Docker image chứa source code, dependencies và model inference cần thiết.
- Không đưa dataset training, raw data, runs, outputs vào image.
- Demo chạy qua HTTPS của Cloud Run để browser cho phép dùng webcam.
- Có health check, benchmark cơ bản, và tài liệu lệnh deploy có thể chạy lại.

Cloud Run phù hợp cho demo public vì nó chạy container, tự cấp HTTPS endpoint, scale theo request, và có thể deploy bằng `gcloud` từ Artifact Registry. Với bài toán realtime webcam, cần xử lý riêng WebSocket timeout, reconnect, cold start và nơi lưu snapshot.

## Kết luận triển khai

Ta sẽ dùng Docker.

Lý do:

- Cloud Run chạy ứng dụng dưới dạng container image.
- Repo hiện đã có `Dockerfile`, `docker-compose.yml`, FastAPI và `/health`.
- Docker giúp môi trường deploy gần giống local hơn so với chạy thủ công.
- Dataset không nên đóng gói vào container; container chỉ cần code + model đã train.

Chiến lược model cho bản demo đầu tiên:

- Đưa `models/best.pt` vào Docker image vì model YOLOv8n hiện nhỏ, dễ deploy và không phụ thuộc bucket lúc startup.
- Giữ `data/`, `runs/`, `outputs/`, zip dataset ở ngoài image.
- Nếu sau này model lớn hơn hoặc có nhiều version, chuyển sang Cloud Storage: container tải model từ `gs://...` khi startup hoặc mount bucket.

## Những gì lấy được từ dự án Cloud Run cũ

Tham khảo repo:

```text
/Users/nguyen_bao/Documents/PTIT/nam4/Cloud/mini-social-network
```

Có thể tái dùng:

- Pattern `deploy-images.sh`: enable API, tạo Artifact Registry repo nếu chưa có, dùng `gcloud builds submit --tag ...`.
- Cách đặt biến `PROJECT_ID`, `REGION`, `REPO_NAME`, `ARTIFACT_HOST`, `REPO_PATH`.
- Quy trình build image bằng Cloud Build thay vì build local.

Không nên bê nguyên:

- Kiến trúc microservice nhiều backend service không cần cho dự án này.
- Dockerfile Node.js không dùng được cho FastAPI/Python.
- Compose nhiều service không cần cho Cloud Run bản đầu tiên.

Repo facemask nên deploy như một Cloud Run service duy nhất:

```text
Browser HTTPS
  -> Cloud Run service
      -> FastAPI
          -> static frontend /
          -> REST /health, /api/v1/predict/image, /api/v1/dataset/capture
          -> WebSocket /api/v1/ws/detect
              -> YOLO model inference
```

## Các vấn đề cần sửa trước khi deploy

### 1. Dockerfile cần đúng chuẩn Cloud Run

Hiện Dockerfile chạy:

```dockerfile
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Cloud Run inject biến môi trường `PORT`; nên sửa Dockerfile để lắng nghe `${PORT:-8080}` thay vì hard-code `8000`.

Đề xuất:

```dockerfile
EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

Không dùng `--reload` trong container production.

### 2. Model đang bị `.dockerignore` loại khỏi image

Hiện `.dockerignore` đang ignore:

```text
*.pt
models/*.pt
```

Nếu chọn đóng gói `models/best.pt` vào Cloud Run image, cần sửa `.dockerignore` theo hướng chỉ allow đúng file deploy:

```text
*.pt
models/*.pt
!models/best.pt
```

Không allow toàn bộ `models/` nếu có nhiều artifact lớn.

Cloud Build cần thêm `.gcloudignore` riêng. `gcloud builds submit` có thể dùng `.gitignore` mặc định để tạo source archive, nên nếu repo ignore `*.pt` thì model sẽ không được upload lên Cloud Build dù `.dockerignore` đã đúng. `.gcloudignore` cũng phải allow đúng file:

```text
*.pt
models/*.pt
!models/
!models/best.pt
```

### 3. Snapshot local không bền trên Cloud Run

Cloud Run filesystem là writable nhưng không persistent. File trong `outputs/snapshots` có thể mất khi instance scale down hoặc restart.

Cho bản demo đầu tiên:

- Có thể tắt snapshot lưu file, chỉ trả kết quả realtime trên UI.
- Hoặc vẫn lưu local để debug trong vòng đời instance, nhưng không coi đó là evidence lâu dài.

Cho bản tốt hơn:

- Upload violation snapshot lên Cloud Storage.
- Lưu metadata vào CSV/JSONL trong Cloud Storage hoặc Firestore.

### 4. WebSocket cần reconnect và timeout

Cloud Run hỗ trợ WebSocket, nhưng WebSocket vẫn bị tính là HTTP request dài. Cần:

- Set request timeout đủ dài, ví dụ 30-60 phút.
- Frontend tự reconnect khi WebSocket đóng.
- Không bật HTTP/2 end-to-end.
- Nếu scale nhiều instance và cần trạng thái chung, phải dùng storage/queue ngoài instance. Với demo hiện tại, mỗi client độc lập nên chưa cần.

### 5. Concurrency phải hiểu theo request trên mỗi instance

`--concurrency` của Cloud Run là số request đồng thời trên một instance, không phải số instance. WebSocket là một request giữ lâu, nên một kết nối WebSocket sẽ chiếm một concurrency slot trong suốt thời gian stream.

Với code hiện tại, mỗi frame được xử lý trực tiếp trong WebSocket handler, chưa có shared queue/batching. Vì vậy bản deploy đầu tiên nên chọn:

```text
--concurrency 1
```

Điều này tránh nhiều WebSocket cùng tranh CPU trong một process. Nhược điểm là client thứ hai có thể bị route sang instance khác, kéo theo cold start và chi phí tăng. Chỉ tăng lên `--concurrency 5` hoặc `--concurrency 10` sau khi đã implement shared inference queue hoặc frame batching.

### 6. Cold start cần được giảm cho demo

PyTorch/Ultralytics load model có thể mất vài giây; cộng thêm cold start của Cloud Run, lần mở demo đầu tiên có thể đứng khá lâu. Với demo phỏng vấn hoặc portfolio, nên giữ một instance warm:

```text
--min-instances 1
```

Thiết lập này phát sinh chi phí idle, nhưng đổi lại URL demo phản hồi ổn định hơn. Nếu chỉ test nhanh và muốn tiết kiệm tuyệt đối, có thể đưa về `--min-instances 0` sau khi demo.

### 7. Webcam remote cần HTTPS

Local `http://localhost:8000` dùng được webcam.

Remote phải dùng HTTPS. Cloud Run tự cấp URL HTTPS dạng:

```text
https://facemask-api-xxxxx.a.run.app
```

Do đó frontend webcam sẽ hợp lý hơn khi chạy từ chính Cloud Run URL, thay vì mở `http://<ip>:8000`.

## Phase triển khai đề xuất

### Phase 0 - Chốt cấu hình deploy

Tạo các biến dùng lại:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="asia-southeast1"
export REPO_NAME="facemask-repo"
export SERVICE_NAME="facemask-compliance"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest"
```

`asia-southeast1` hợp lý cho demo ở Việt Nam/Singapore. Nếu project GCP cũ đang dùng region khác thì có thể giữ cùng region để dễ quản lý.

### Phase 0.5 - Benchmark local trước khi deploy

Trước khi đưa lên Cloud Run, cần đo baseline local để sau này so sánh Cloud Run CPU/GPU có ý nghĩa.

Tối thiểu cần ghi:

- Local Mac CPU: FPS, average latency, p95 latency.
- Local Docker CPU: FPS, average latency, p95 latency.
- Vast.ai hoặc local GPU nếu còn máy: FPS, average latency, p95 latency.

Ví dụ:

```bash
python scripts/benchmark.py \
  --model models/best.pt \
  --images path/to/test_images \
  --imgsz 640
```

Script hiện tại benchmark theo thư mục ảnh và đã có average latency, p95 latency, FPS. Nếu muốn benchmark video hoặc remote HTTP endpoint thì bổ sung thêm mode riêng trước khi ghi số vào README/CV. Không so sánh Cloud Run nếu chưa có baseline local.

### Phase 1 - Làm container chạy được trên Cloud Run

Việc cần làm:

- Sửa Dockerfile dùng biến `PORT`.
- Đảm bảo `MODEL_PATH=models/best.pt` tồn tại trong image hoặc được truyền qua env.
- Cân nhắc tạo `requirements-cloudrun.txt` dùng `opencv-python-headless` thay vì `opencv-python` để image gọn và tránh dependency GUI.
- Đảm bảo `/health` không load model nặng.
- Sau khi chỉnh `.dockerignore` và `.gcloudignore`, verify thật sự model nằm trong image vì negation `!` cần test bằng build thực tế.
- Build local test trước:

```bash
docker build -t facemask-cloudrun .
docker run --rm --entrypoint sh facemask-cloudrun -lc 'ls -lh models/best.pt'
docker run --rm -p 8080:8080 -e PORT=8080 -e MODEL_PATH=models/best.pt facemask-cloudrun
curl http://localhost:8080/health
```

Kết quả đạt:

- `curl /health` trả `{"status":"ok"}`.
- Lệnh `ls -lh models/best.pt` trong container thấy đúng file model.
- Mở `http://localhost:8080` thấy UI.
- Test upload image hoặc webcam local chạy được.

### Phase 2 - Tạo script build và push giống dự án cũ

Tạo script mới, ví dụ:

```text
scripts/deploy_cloudrun.sh
```

Nội dung logic lấy từ `mini-social-network/deploy-images.sh`:

```bash
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com --project "$PROJECT_ID"

gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Docker repository for facemask compliance demo" \
  --project="$PROJECT_ID" \
  --quiet || true

gcloud builds submit . \
  --tag "$IMAGE" \
  --project "$PROJECT_ID"
```

Khác với dự án microservice cũ:

- Chỉ build một image.
- Không cần loop qua nhiều service.
- Không cần API gateway riêng.

### Phase 3 - Deploy Cloud Run CPU bản đầu tiên

Deploy trước bằng CPU để có public demo nhanh:

```bash
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 1 \
  --min-instances 1 \
  --timeout 3600 \
  --set-env-vars MODEL_PATH=models/best.pt,APP_CONFIG=configs/app.yaml \
  --project "$PROJECT_ID"
```

Giải thích:

- `--timeout 3600`: cho WebSocket sống tối đa 60 phút.
- `--concurrency 1`: phù hợp với code hiện tại vì chưa có batching/shared inference queue.
- `--min-instances 1`: giữ một instance warm để giảm cold start lúc mở demo.
- `--memory 4Gi`: đủ an toàn hơn cho PyTorch/Ultralytics so với memory thấp.
- `--allow-unauthenticated`: cần cho demo public.

Nếu muốn tiết kiệm chi phí sau khi demo xong:

```bash
gcloud run services update "$SERVICE_NAME" \
  --region "$REGION" \
  --min-instances 0 \
  --project "$PROJECT_ID"
```

Sau deploy:

```bash
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format 'value(status.url)')

curl "$SERVICE_URL/health"
open "$SERVICE_URL"
```

Kết quả đạt:

- URL public mở được.
- Browser xin quyền camera.
- WebSocket dùng `wss://.../api/v1/ws/detect`.
- UI có FPS/latency/detection.

### Phase 4 - Sửa frontend cho Cloud Run nếu cần

Frontend nên tự xác định WebSocket URL theo origin hiện tại:

```javascript
const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/ws/detect`;
```

Cần thêm reconnect:

- Khi WebSocket close, đợi 1-2 giây rồi connect lại.
- Khi timeout Cloud Run ngắt sau 60 phút, client không chết hẳn.
- UI hiển thị trạng thái `Reconnecting` thay vì đứng yên.

### Phase 4.5 - Test WebSocket qua HTTPS

Sau khi deploy, bắt buộc test browser thật trên Cloud Run URL:

```text
https://<service-id>.<region>.run.app
```

Checklist test:

- Mở URL bằng HTTPS, không dùng IP hoặc HTTP.
- Browser cấp quyền camera thành công.
- DevTools Console không có mixed content warning.
- WebSocket URL là `wss://.../api/v1/ws/detect`, không phải `ws://...`.
- Network tab thấy WebSocket status `101 Switching Protocols`.
- Tắt mạng hoặc reload tab, frontend reconnect được.
- Server log có connection open/close bình thường, không có ASGI error.

### Phase 5 - Benchmark và ghi số thật

Chạy benchmark với Cloud Run URL:

```bash
python scripts/benchmark.py \
  --url "$SERVICE_URL/api/v1/predict/image" \
  --image path/to/test.jpg \
  --requests 50
```

Nếu script hiện tại chưa hỗ trợ remote URL thì bổ sung.

Cần ghi lại:

- Cold start time.
- Average latency.
- p95 latency.
- FPS webcam thực tế trên Cloud Run CPU.
- Số client đồng thời test được trước khi latency quá cao.

Không ghi vào CV các số chưa đo.

### Phase 6 - Quyết định có cần Cloud Run GPU không

Cloud Run hiện có hỗ trợ GPU cho service, ví dụ NVIDIA L4. Tuy nhiên đây không phải bước đầu tiên vì:

- Cần quota GPU theo region.
- Cần image/dependency CUDA/PyTorch phù hợp.
- Cost cao hơn CPU.
- Với YOLOv8n và demo webcam 1-2 client, CPU có thể đủ để chứng minh project.

Nếu CPU quá chậm hoặc muốn demo nhiều client:

```bash
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 4 \
  --memory 16Gi \
  --timeout 3600 \
  --project "$PROJECT_ID"
```

Khi dùng GPU, cần kiểm tra lại:

- PyTorch có CUDA trong image hay không.
- Ultralytics detect được `device=0` hay không.
- Cloud Run GPU quota trong region.
- Chi phí theo thời gian instance active, đặc biệt khi WebSocket giữ connection lâu.

## Những gì không deploy

Không đưa vào Docker image:

```text
data/raw/
data/mask_detection/
runs/
outputs/
*.zip
mask_yolov8n_pwmfd_baseline/
```

Không upload dataset lên Cloud Run. Dataset chỉ dùng cho training/audit. Cloud Run chỉ cần model đã train và code inference.

## Kết quả triển khai hiện tại

Ngày triển khai: 2026-05-11.

Cloud Run service:

```text
Project: healthy-system-479516-u4
Region: asia-southeast1
Service: facemask-compliance
Primary URL: https://facemask-compliance-585107925400.asia-southeast1.run.app
Run URL: https://facemask-compliance-szbjef7jsa-as.a.run.app
Image: asia-southeast1-docker.pkg.dev/healthy-system-479516-u4/facemask-repo/facemask-compliance:fbdc681-20260511220601
Revision: facemask-compliance-00002-xvj
```

Runtime config:

```text
CPU: 2
Memory: 4Gi
Concurrency: 1
Min instances: 1
Timeout: 3600s
PRELOAD_MODEL: 1
MODEL_PATH: models/best.pt
```

Smoke test đã chạy:

```text
GET /health: OK
GET /: HTML frontend OK
POST /api/v1/predict/image: OK
WSS /api/v1/ws/detect: OK
```

Latency quan sát được với một ảnh test:

```text
Trước preload:
- REST first inference: ~15262 ms
- WSS first inference: ~16445 ms
- Warm REST: ~300 ms
- Warm WSS: ~233 ms

Sau preload + shared predictor:
- REST first inference sau deploy: ~1276 ms
- WSS first inference sau deploy: ~1190 ms
- Warm REST: ~151 ms
- Warm WSS: ~127 ms
```

Ghi chú: `min-instances=1` đang bật để giữ demo warm. Sau khi không cần demo public nữa, có thể giảm chi phí bằng:

```bash
gcloud run services update facemask-compliance \
  --region asia-southeast1 \
  --min-instances 0 \
  --project healthy-system-479516-u4
```

## Checklist hoàn thành

- [x] Dockerfile dùng `${PORT:-8080}` và không dùng reload.
- [x] Model deploy strategy rõ ràng: image hoặc Cloud Storage.
- [x] `.dockerignore` không vô tình loại mất model deploy.
- [x] `.gcloudignore` không vô tình loại mất `models/best.pt` khỏi Cloud Build source archive.
- [x] Đã verify `models/best.pt` tồn tại bên trong Docker image.
- [x] `docker build` và `docker run` local pass.
- [ ] Có benchmark baseline local trước khi deploy.
- [x] Artifact Registry repo được tạo.
- [x] Cloud Build build image thành công.
- [x] Cloud Run deploy thành công.
- [x] Cloud Run deploy có cấu hình concurrency phù hợp với code hiện tại.
- [x] `--min-instances 1` được bật trong giai đoạn demo, hoặc ghi rõ lý do tắt.
- [x] `/health` public pass.
- [x] Frontend HTML được phục vụ từ HTTPS Cloud Run URL.
- [x] WebSocket từ script chạy qua `wss://` Cloud Run URL.
- [ ] Không có mixed content warning.
- [ ] Camera permission granted trên HTTPS.
- [ ] WebSocket reconnect được khi bị close.
- [ ] Benchmark latency/p95/FPS được ghi vào report.
- [ ] README/DEPLOYMENT cập nhật URL demo và lệnh deploy.

## Tài liệu tham khảo chính thức

- Cloud Run deploy container từ Artifact Registry: https://cloud.google.com/artifact-registry/docs/integrate-cloud-run
- Cloud Run container port và biến `PORT`: https://cloud.google.com/run/docs/configuring/services/containers
- Cloud Run WebSocket: https://cloud.google.com/run/docs/triggering/websockets
- Cloud Run concurrency: https://cloud.google.com/run/docs/about-concurrency
- Cloud Run minimum instances: https://cloud.google.com/run/docs/configuring/min-instance
- Cloud Run request timeout: https://cloud.google.com/run/docs/configuring/request-timeout
- Cloud Run GPU services: https://cloud.google.com/run/docs/configuring/services/gpu
- Cloud Run filesystem không persistent: https://cloud.google.com/run/docs/overview/what-is-cloud-run
