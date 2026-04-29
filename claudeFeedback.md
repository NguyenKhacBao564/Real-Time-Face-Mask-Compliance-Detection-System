# Claude Feedback — Real-Time Face Mask Compliance Detection System

> Đối tượng đánh giá: portfolio dùng để **ứng tuyển vị trí Intern AI/ML Engineer**.
> Tài liệu được giữ ngắn gọn, tập trung vào những điều thực sự ảnh hưởng đến kết quả tuyển dụng intern.

---

## 1. Tóm tắt nhanh

Dự án **đủ tốt cho mục tiêu intern** ngay cả ở phạm vi MVP hiện tại. Một intern không cần novelty cao — cần chứng minh:

1. Hiểu pipeline ML end-to-end (data → train → deploy).
2. Viết code sạch, có cấu trúc.
3. Đo lường trung thực, biết debug.
4. Trình bày kết quả rõ ràng.

Dự án này đáp ứng cả 4 yêu cầu trên. Phần cần đầu tư nhất là **dataset** vì đó là rào cản kỹ thuật lớn nhất của bài toán 3-class mask compliance.

---

## 2. Điểm mạnh

- **Phạm vi MVP rõ ràng và có kỷ luật**
  Tách rõ Required / Optional / Out-of-scope. Hiếm gặp ở project intern, thường bị "scope creep".

- **Kiến trúc thực dụng**
  Single-stage YOLOv8n thay vì 2-stage; WebSocket trả JSON detections thay vì base64 frame; chọn baseline nhỏ trước khi nâng cấp. Đây là các quyết định của người đã đọc tài liệu kỹ.

- **Tư duy dataset chín chắn**
  Phân tầng V1 → V2 → V3, cảnh báo không merge mù 2-class + 3-class, có audit checklist. Đa số intern không nghĩ đến những điều này.

- **Bộ metric đầy đủ**
  mAP@50, mAP@50:95, precision/recall theo lớp, FPS, latency p95, model size — đủ để viết bảng kết quả thuyết phục.

- **MLOps/Git hygiene tốt**
  Không commit weights, dùng Vast.ai cho training, lưu metrics + config trong repo. Cho thấy hiểu quy trình thực tế.

- **Có lộ trình theo tuần với deliverable cụ thể**
  Người đọc CV/repo biết ngay bạn làm gì mỗi tuần.

---

## 3. Điểm yếu

- **Lớp `incorrect_mask` là điểm chết kỹ thuật**
  Dataset 3-class hiện đang nhắm tới (`andrewmvd/face-mask-detection`) chỉ có **123 instance** `mask_weared_incorrect` trên 853 ảnh. Train xong precision/recall lớp này gần như chắc chắn sẽ **rất thấp** (mAP < 0.3). Đây là rủi ro lớn nhất của dự án và `Dataset.md` chưa nêu kế hoạch cụ thể để xử lý.

- **Định nghĩa "real-time" còn mơ hồ**
  Bao nhiêu FPS? Trên hardware nào? CPU laptop hay GPU? Cần ngưỡng cụ thể (ví dụ ≥ 15 FPS @ 640x480 trên CPU Intel i5) để có target rõ ràng khi benchmark.

- **Thiếu baseline so sánh**
  Một con số mAP@50 = 0.85 không có ý nghĩa nếu không so với gì. Nên có baseline: YOLOv8n pretrain (zero-shot) trên val set, hoặc một public repo phổ biến.

- **Frontend mô tả quá sơ sài**
  Chỉ "canvas bounding boxes". Demo video sẽ thiếu sức nặng nếu không có FPS counter, latency display, violation log UI.

- **Thiếu mục Privacy & Ethics**
  Detection trên gương mặt người — kể cả không nhận dạng — vẫn nên có 5-10 dòng về privacy considerations. Recruiter nghiêm túc (đặc biệt outsource cho EU/Mỹ) hay hỏi.

- **Không có gì để chống "vendor lock-in" với Ultralytics**
  Toàn bộ training code phụ thuộc vào `ultralytics` package. Không sai cho intern, nhưng nếu hiểu cách export ONNX và load lại bằng `onnxruntime` cho inference thì sẽ rất ấn tượng.

---

## 4. Đánh giá MaskedFace-Net (cabani/MaskedFace-Net)

### Tóm tắt dataset

- 137,016 ảnh face crop dựa trên FFHQ (~1024×1024).
- 2 lớp: Correctly Masked Face (CMFD) ~67k và Incorrectly Masked Face (IMFD) ~67k.
- Sinh học bằng cách paste mask lên face → **synthetic**, không phải ảnh thật.
- Dung lượng ~38 GB tổng.
- License: **Creative Commons BY-NC-SA 4.0** (NonCommercial, ShareAlike).

### Có dùng cho dự án này được không?

**Không phù hợp làm dataset chính.** Lý do:

| Vấn đề | Chi tiết |
| --- | --- |
| **Sai dạng bài toán** | Là ảnh face crop (đã cắt sẵn), không có bounding box, không có scene context. YOLO cần ảnh full-scene + bbox annotations. |
| **Thiếu lớp `no_mask`** | Chỉ có correct và incorrect. Không phục vụ được bài toán 3-class. |
| **Synthetic** | Mask được paste bằng đồ họa, không giống mask thật trong webcam frame → domain gap. |
| **License NC-SA** | Non-Commercial. Nếu sau này muốn đưa vào CV/portfolio public hoặc demo cho công ty, phải attribution và không được dùng cho mục đích thương mại. Phiền phức. |
| **Quá lớn** | 38 GB cho một MVP intern là overkill, tốn thời gian download trên Vast.ai. |

### Khi nào thì MaskedFace-Net hữu ích?

- **Pretrain một classifier** correct vs incorrect cho pipeline 2-stage (face detector → classifier) — đây là hướng V4/research.
- **Nguồn copy-paste augmentation**: cắt face IMFD → paste vào ảnh scene-level (từ FMLD/PWMFD) để tăng số lượng `incorrect_mask` instance. Kỹ thuật này gọi là *Copy-Paste Augmentation* (Ghiasi et al., 2021).
- **Không phải task hiện tại của intern.** Ghi chú vào "Future Work".

**Kết luận**: Giữ nguyên quan điểm trong `Dataset.md` — chỉ tham chiếu MaskedFace-Net như tương lai, không dùng cho MVP.

---

## 5. Dataset đề xuất (giải quyết vấn đề `incorrect_mask`)

Đây là phần quan trọng nhất. Đề xuất theo thứ tự ưu tiên:

### 5.1. **PWMFD — Properly Wearing Masked Face Detection Dataset** ⭐ (Top pick)

- **Repo**: <https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset>
- **Quy mô**: 9,205 ảnh, 18,532 labeled instances
- **Classes**: `face_with_mask`, `face_with_mask_incorrect`, `face_without_mask` — **đúng 3 lớp dự án cần**
- **Format**: PASCAL VOC XML (chuyển sang YOLO bằng script đơn giản)
- **Ưu điểm**:
  - Scene-level images (đa người, đa góc) — đúng dạng YOLO detection.
  - Đã có sẵn lớp `incorrect_mask` dồi dào hơn andrewmvd nhiều.
  - Quy mô vừa phải, train trên Vast.ai 1× T4 trong vài giờ.
- **Nhược điểm**: Chất lượng annotation không đồng đều (cần audit), một số ảnh trùng lặp nguồn.

### 5.2. **FMLD — Face Mask Label Dataset** ⭐ (Best for `incorrect_mask`)

- **Repo**: <https://github.com/borutb-fri/FMLD>
- **Paper**: [How to Correctly Detect Face-Masks for COVID-19 (Applied Sciences, 2021)](https://www.mdpi.com/2076-3417/11/5/2070)
- **Quy mô**: 41,934 face annotations từ MAFA + WIDER Face
- **Classes**: correctly worn / incorrectly worn / without mask — **đúng 3 lớp**
- **Format**: CSV annotations với bbox, gender, ethnicity, pose; cần script chuyển sang YOLO
- **Ưu điểm**:
  - Dataset **lớn nhất và chất lượng cao nhất** cho 3-class mask detection trong tự nhiên (in the wild).
  - Chỉ là annotation file — bạn tải MAFA và WIDER Face riêng (free), nên repo nhẹ.
  - Có paper tham chiếu → trích dẫn được trong README.
- **Nhược điểm**:
  - Phải tải 2 dataset gốc (MAFA cần đăng ký).
  - Pipeline tiền xử lý phức tạp hơn (gộp 2 nguồn, viết converter).

### 5.3. **andrewmvd/face-mask-detection** (Kaggle) — Backup nhanh

- **Link**: <https://www.kaggle.com/datasets/andrewmvd/face-mask-detection>
- **Quy mô**: 853 ảnh, ~4,000 instances; trong đó `mask_weared_incorrect` chỉ ~123 instance
- **Vai trò trong dự án**: chỉ dùng để **prototype pipeline tuần 1** (verify training chạy được), không dùng làm dataset chính.

### 5.4. **AIZOO FaceMaskDetection** — Backup cho 2-class baseline

- **Repo**: <https://github.com/AIZOOTech/FaceMaskDetection>
- **Quy mô**: 7,959 ảnh từ WIDER + MAFA, 2 classes (`face`, `face_mask`)
- **Vai trò**: dataset 2-class chất lượng cao, dùng cho V1 baseline thay cho `parot99` nếu muốn quy mô lớn hơn.

---

## 6. Chiến lược dataset đề xuất (thay thế kế hoạch hiện tại)

Thay cho V1/V2/V3 trong `Dataset.md` hiện tại, đề xuất pipeline sau:

```text
Stage 1 — Smoke test (1-2 ngày)
  Dataset: andrewmvd/face-mask-detection (853 ảnh)
  Mục tiêu: verify pipeline train → val → predict chạy đầu cuối
  Không quan tâm metric, chỉ cần loss giảm.

Stage 2 — Baseline thật (V1)
  Dataset: PWMFD (9,205 ảnh, 3 classes)
  Mục tiêu: train YOLOv8n từ pretrain COCO
  Output: bộ model đầu tiên có metric "có thể trình bày"
  Kỳ vọng: mAP@50 ≈ 0.75-0.85 trên correct/no_mask, 0.40-0.55 trên incorrect

Stage 3 — Cải thiện incorrect_mask (V2)
  Dataset: PWMFD + FMLD merge có chiến lược
  Kỹ thuật bổ sung:
    - Class weights / weighted sampler ưu tiên incorrect_mask
    - Augmentation mạnh trên incorrect_mask: mosaic, copy-paste, mixup
    - (Optional) copy-paste face crops từ MaskedFace-Net IMFD
  Mục tiêu: tăng recall lớp incorrect_mask lên ≥ 0.65

Stage 4 — Real-world test (V3)
  Tự thu 100-200 webcam frame của bản thân + bạn bè
  CHỈ DÙNG ĐỂ EVAL, không train
  Mục đích: kiểm tra domain gap webcam vs dataset
```

---

## 7. Đề xuất cải tiến khác (theo thứ tự ưu tiên cho intern)

### Ưu tiên cao (nên làm)

- **Cụ thể hóa SLA real-time**: viết ngưỡng FPS và latency cụ thể trên hardware cụ thể trong README.
- **Bổ sung baseline so sánh**: zero-shot YOLOv8n trên val set ở Tuần 1.
- **Demo video 60 giây**: deliverable hạng A. Recruiter xem video trước khi đọc README. Cần có FPS counter, bbox màu theo lớp, tiếng (optional).
- **Bổ sung mục Privacy & Ethics ngắn** trong README (5-10 dòng).
- **Honest results table**: nếu `incorrect_mask` precision/recall thấp, **viết ra trung thực** kèm phân tích nguyên nhân (data imbalance) + hướng khắc phục. Recruiter intern đánh giá rất cao sự trung thực + tư duy phân tích, hơn là số đẹp.

### Ưu tiên trung bình (làm nếu có thời gian)

- **FPS counter & latency overlay** trên frontend canvas.
- **Snapshot logging**: lưu frame có vi phạm vào `outputs/violations/` với timestamp.
- **README có bảng kết quả per-class** (precision/recall từng lớp), không chỉ tổng.

### Ưu tiên thấp (bỏ qua nếu trễ deadline)

- ONNX export + benchmark.
- Edge deployment (Raspberry Pi, Jetson).
- Compliance dashboard.

> Với mục tiêu **intern**, không cần các "wow factor" này. Hoàn thành vững phần MVP + báo cáo trung thực đã đủ ấn tượng.

---

## 8. Cảnh báo cuối

- **Đừng gian lận metric**. Một con số mAP@50 = 0.95 nghi ngờ sẽ bị recruiter phát hiện qua câu hỏi technical (ví dụ "test set có overlap với train không?", "bạn đo trên split nào?"). Trung thực là tài sản lớn nhất của profile intern.
- **Đừng cố bao nhiêu thứ**. Hoàn thành 1 thứ tốt > 5 thứ dang dở. Recruiter intern không kỳ vọng bạn có ONNX + Jetson + dashboard. Họ kỳ vọng bạn hiểu được mình đang làm gì.
- **Viết README để người không quen dự án vẫn đọc được trong 5 phút**. Recruiter chỉ có 5 phút.

---

## 9. CV bullet đề xuất (cho mục tiêu intern)

```text
Built an end-to-end real-time face mask compliance detection system:
fine-tuned YOLOv8n on PWMFD + FMLD for 3-class detection
(correct/incorrect/no mask), deployed via FastAPI WebSocket and Docker
with a browser webcam frontend. Achieved [X] mAP@50, [Y] FPS on CPU,
and documented per-class limitations on the underrepresented
"incorrect mask" class.
```

Câu cuối ("documented per-class limitations…") là điểm cộng — cho thấy bạn hiểu data và biết phân tích, không chỉ chạy `model.train()`.
