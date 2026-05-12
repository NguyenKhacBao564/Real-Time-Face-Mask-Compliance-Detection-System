# Gameloft AI Engineer Intern — Interview Preparation

> Vị trí: **AI Engineer Intern (Game Development)** — Gameloft Saigon
> Trọng tâm: **Dataset quality & annotation tại quy mô lớn** cho các dự án CV (Applaydu là dự án rõ nhất từ JD).
> Hồ sơ ứng viên: 3 dự án CV cá nhân (FallGuard Labeling Tool, Face Mask Compliance, Traffic Violation Detection).

---

## 0. Bối cảnh cần nắm trước khi đi

- **Applaydu** là app trẻ em (3–9 tuổi) của Gameloft × Ferrero (Kinder Surprise) — quét đồ chơi vật lý bằng camera điện thoại để mở khóa nội dung số. Bài toán JD nói (~1.000 real-world objects) chính là task scan đồ chơi này.
- **Thách thức thực tế** của Applaydu: occlusion (tay trẻ em che đồ chơi), motion blur (rung tay), lighting (phòng khách/ngủ), reflection (đồ chơi nhựa bóng).
- Vì là **intern dataset**, recruiter quan tâm nhất:
  1. Bạn có **kiên nhẫn** với việc lặp đi lặp lại không?
  2. Bạn có **detail-oriented** không?
  3. Bạn có **hiểu chất lượng data quan trọng hơn model fancy** không?
  4. Bạn có khả năng **document có hệ thống** không?
- Đừng cố tỏ ra "AI guru". Họ không tuyển Senior ML Engineer.

---

## 1. Vòng 1: Screening Call (HR/Recruiter)

**Hình thức**: gọi điện hoặc Google Meet, ~15–30 phút, tiếng Việt (đôi khi đan xen English để test).
**Mục đích**: lọc cơ bản — bạn có phù hợp về thời gian, kỳ vọng lương, level English, motivation.

### 1.1. Câu hỏi thường gặp

#### Q1. *"Anh/em có thể giới thiệu một chút về bản thân và lý do ứng tuyển vị trí này không?"*

**Ý đồ HR**: nghe pitch 60–90 giây để đánh giá khả năng giao tiếp + sự chuẩn bị.

**Khung trả lời (Việt)**:
> "Em là Bảo, sinh viên năm 4 ngành Công nghệ Thông tin tại PTIT, đang tập trung học sâu về Computer Vision. Trong 1 năm gần đây em đã hoàn thành 3 dự án cá nhân xoay quanh chủ đề chuẩn bị và đảm bảo chất lượng dataset cho các bài toán detection — gồm dataset assistant tự động cho gán nhãn pose, hệ thống phát hiện đeo khẩu trang, và pipeline phát hiện vi phạm giao thông kèm OCR biển số. Em ứng tuyển vị trí này vì cả 3 dự án đều khiến em nhận ra: model performance phần lớn đến từ chất lượng data, và em muốn được làm điều đó ở quy mô thật — đặc biệt với một dự án thực tế nhiều thách thức như Applaydu."

**Lưu ý**: nhắc Applaydu là điểm cộng (cho thấy đã research). Đừng đọc thuộc lòng — luyện 3–4 lần đến khi nói tự nhiên.

---

#### Q2. *"Em biết gì về Gameloft và vị trí này?"*

**Khung**:
> "Gameloft Saigon là studio lớn nhất của Gameloft toàn cầu, đứng sau các tựa game như Asphalt Legends Unite, Dungeons & Dragons, Modern Combat. Vị trí này theo em hiểu là intern hỗ trợ team AI/Data trong việc xây dataset CV chất lượng cao — quy mô khoảng 1.000 real-world objects — phục vụ object recognition, mà em đoán là cho các dự án scan-toy như Applaydu Season 6."

---

#### Q3. *"Em có thể cam kết bao nhiêu ngày/tuần và trong bao lâu?"*

**Lưu ý**: lương intern Gameloft tính theo committed working days (từ 6tr gross/tháng). Trả lời rõ ràng, không lưỡng lự.

**Khung**:
> "Em có thể cam kết **[3/4/5] ngày/tuần** trong tối thiểu **[3/6] tháng**, có thể bắt đầu từ **[ngày cụ thể]**. Trên trường em vẫn còn vài môn nhưng đã sắp lịch để full availability vào các ngày làm việc."

**Đừng** trả lời "tùy công ty" — HR không thích.

---

#### Q4. *"Mức lương em mong đợi?"*

**JD ghi từ 6tr gross/month**. Đừng đòi cao hơn — họ có thang cố định cho intern.

**Khung**:
> "Em hiểu mức intern allowance của Gameloft là từ 6 triệu gross dựa trên committed days. Em ổn với mức theo policy của công ty."

---

#### Q5. *"Em đánh giá tiếng Anh của mình ở mức nào?"*

**Đừng** nói "fluent" nếu không thật — họ sẽ test ngay sau đó. Trung thực:

**Khung**:
> "Em đọc tài liệu kỹ thuật tốt — paper, doc, GitHub README. Nói thì em ở mức intermediate, cần luyện thêm về fluency nhưng đủ để giao tiếp công việc và meeting với team kỹ thuật."

**Khả năng cao** HR sẽ ngay lập tức nói "Can you switch to English for a bit?" → chuẩn bị sẵn 30 giây giới thiệu bản thân bằng tiếng Anh.

---

#### Q6. *"Em đang phỏng vấn ở đâu khác không?"*

**Khung trung thực**:
> "Em đang nộp một vài chỗ về AI/CV intern, nhưng Gameloft là ưu tiên hàng đầu vì quy mô dataset và scope sản phẩm thực tế."

---

#### Q7. *"Em có câu hỏi nào cho recruiter không?"*

**Luôn luôn có câu hỏi**. Một câu là đủ. Tránh hỏi lương/leave (đã có trong JD).

**Gợi ý**:
- "Anh/chị có thể chia sẻ team em sẽ làm cùng có khoảng bao nhiêu người, và workflow daily như thế nào không?"
- "Sau khi pass, lộ trình phát triển từ intern → fulltime ở Gameloft thường diễn ra như thế nào?"
- "Vòng Test sắp tới sẽ tập trung vào skill nào để em chuẩn bị tốt hơn?"

---

## 2. Vòng 2: Test

**Hình thức dự đoán**: take-home Python task (1–3 ngày) hoặc onsite/online test 60–120 phút. Trọng tâm: **dataset processing, không phải model training**.

### 2.1. Các dạng bài có khả năng cao

#### Dạng 1: Dataset Audit Script (xác suất rất cao)

**Đề mẫu**:
> "Cho một thư mục dataset với cấu trúc YOLO (`images/`, `labels/`). Viết một script Python in ra: số ảnh mỗi split, số label mỗi class, số file label rỗng, số bbox có tọa độ ngoài [0, 1], số ảnh không có label tương ứng, và ngược lại."

**Cách chuẩn bị**: bạn đã có sẵn code này trong `scripts/audit_dataset.py` của dự án Face Mask. Học thuộc cấu trúc, biết viết lại từ đầu trong 30 phút.

**Skeleton để luyện**:
```python
from collections import Counter
from pathlib import Path

def audit(dataset_dir: Path):
    report = {}
    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        images = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".jpg",".png",".jpeg"}}
        labels = {p.stem for p in lbl_dir.glob("*.txt")}

        class_counts = Counter()
        invalid_boxes = 0
        empty_labels = 0
        for lbl in lbl_dir.glob("*.txt"):
            lines = lbl.read_text().strip().splitlines()
            if not lines: empty_labels += 1
            for line in lines:
                parts = line.split()
                if len(parts) != 5: invalid_boxes += 1; continue
                cls, *coords = parts
                if any(not (0 <= float(c) <= 1) for c in coords):
                    invalid_boxes += 1
                else:
                    class_counts[int(cls)] += 1

        report[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": list(images - labels)[:5],
            "orphan_labels": list(labels - images)[:5],
            "empty_labels": empty_labels,
            "invalid_boxes": invalid_boxes,
            "class_counts": dict(class_counts),
        }
    return report
```

#### Dạng 2: Annotation Format Conversion

**Đề mẫu**:
> "Chuyển một dataset từ PASCAL VOC XML sang YOLO format. Xử lý case bbox bị clip ra ngoài ảnh, label name không khớp class map, file XML thiếu width/height."

**Chuẩn bị**: học thuộc `scripts/convert_dataset.py` của Face Mask project — bạn đã có giải pháp tốt với class normalization, bbox clipping, deterministic naming.

#### Dạng 3: Duplicate / Near-Duplicate Detection

**Đề mẫu**:
> "Cho 1000 ảnh, viết script phát hiện ảnh trùng lặp (exact + near-duplicate). Trả về danh sách cluster."

**Approaches**:
- Exact: SHA-1/MD5 hash của file bytes.
- Near-dup: perceptual hash (`imagehash` library, pHash/dHash) → khoảng cách Hamming.
- Nếu được dùng deep features: ResNet/CLIP embedding + cosine similarity.

```python
import imagehash
from PIL import Image
from collections import defaultdict

clusters = defaultdict(list)
for path in image_dir.glob("*.jpg"):
    h = imagehash.phash(Image.open(path))
    clusters[str(h)].append(path)
duplicates = [v for v in clusters.values() if len(v) > 1]
```

#### Dạng 4: Quality Check Heuristics

**Đề mẫu**:
> "Phát hiện ảnh quá tối, quá mờ, hoặc bbox quá nhỏ trong dataset."

**Tools**:
- Tối: trung bình kênh sáng (`np.mean(gray)`) < threshold.
- Mờ: variance of Laplacian (`cv2.Laplacian(gray, cv2.CV_64F).var()`) < threshold.
- Bbox nhỏ: `w * h < min_area` (sau khi denormalize).

#### Dạng 5: Logic câu hỏi conceptual (multiple choice / short answer)

Dự đoán có:
- Difference between classification, detection, segmentation.
- Why train/val/test split, data leakage examples.
- What is class imbalance, mitigation strategies.
- Annotation quality metrics (inter-annotator agreement, IoU consistency).

### 2.2. Tips chung khi làm Test

- **Đọc đề kỹ 2 lần** trước khi viết code.
- **Edge cases là trọng số chính**: file rỗng, file bị corrupt, bbox negative, class id ngoài phạm vi, MacOS hidden files (`.DS_Store`, `__MACOSX`).
- **Document code**: comment ngắn gọn lý do, viết 1 README giải thích cách chạy + decisions.
- **In ra report có cấu trúc** (JSON/YAML), không print loạn.
- **Reproducibility**: dùng `random.seed`, ghi rõ Python version + dependencies.
- Nếu cho deadline 3 ngày → **giao trước deadline 12 tiếng** để show tính kỷ luật.

---

## 3. Vòng 3: Interview (Technical + Behavioral)

**Hình thức**: onsite hoặc Google Meet, 45–75 phút, có thể có 1–2 AI Engineer + 1 HR. Có thể dùng cả Việt và Anh.

### 3.1. Phần A — Project Deep-Dive (~25 phút)

Họ sẽ chọn 1–2 project trong CV và đào sâu. **Project Face Mask** và **FallGuard** có khả năng cao bị hỏi nhất vì gần với JD.

#### Câu hỏi mẫu cho FallGuard Labeling Tool

**Q**: *"Tại sao em tự build labeling tool thay vì dùng CVAT, Label Studio, Roboflow?"*

**Khung trả lời tốt**:
> "Em có thử Label Studio và CVAT trước. Vấn đề là use case của em là pose annotation từ video — cần auto-suggest 13 keypoint từ YOLO-Pose và cho người review *sửa nhanh từng keypoint*, kèm action 'uncertain' và 'exclude'. Các tool có sẵn hỗ trợ keypoint nhưng UX cho human-in-the-loop review không phù hợp với batch của em. Streamlit cho phép em prototype UI trong 1 ngày + tích hợp pipeline Python sẵn có. Tradeoff là không scale được nhiều annotator song song — đó là điểm em sẽ giải quyết bằng tool có sẵn nếu làm ở scale công ty."

**Bài học**: thừa nhận giới hạn của lựa chọn của mình → cho thấy tư duy chín chắn.

**Q**: *"656 frame có ít không? Sao không tự label hết bằng tay luôn?"*

**Khung**:
> "656 frame là sau khi review, không phải tổng số frame raw. Pipeline em là: 5 video → ~6.000 frame raw → frame extraction theo motion delta → auto pose label → human review → reviewed-only export = 656 frame chất lượng cao. Cách này cho ra dataset clean hơn label hết bằng tay vì auto-suggest tránh bias annotator + frame extraction loại các frame trùng lặp gần nhau."

**Q**: *"QA report của em có những metric gì?"*

**Trả lời cụ thể**:
- Per-class instance count.
- Per-frame số keypoint missing.
- Confidence distribution của auto-label trước review.
- Số frame được mark "uncertain" / "exclude".
- Inter-version diff giữa 2 version snapshot.

#### Câu hỏi mẫu cho Face Mask Project

**Q**: *"mAP@50 = 0.970 nghe rất cao. Em có chắc không leak data?"*

**Khung trả lời (RẤT QUAN TRỌNG)**:
> "Em có cân nhắc khả năng leak. Cụ thể em làm 3 bước: (1) split bằng SHA-1 hash của filename, deterministic và độc lập với thứ tự file; (2) verify rằng val/test không chứa filename xuất hiện trong train; (3) check thủ công 50 ảnh val để chắc chắn không có cùng person/cùng frame chỉ khác augmentation. Nhưng em vẫn nghi ngờ con số 0.970 vì PWMFD có nhiều ảnh từ cùng identity ở các góc khác nhau — đó là 1 source of leakage em chưa loại trừ hoàn toàn. Nếu em làm lại em sẽ split theo identity (nếu có metadata), không phải theo file."

**→ Trả lời này SHOW thái độ trung thực + tư duy phản biện** — đây là điểm hay nhất bạn có thể tạo trong cuộc phỏng vấn.

**Q**: *"Class incorrect_mask của em performance thế nào?"*

**Khung**:
> "Thấp hơn 2 class còn lại đáng kể — recall khoảng [con số thực bạn đo được]. Nguyên nhân: PWMFD có ~[N]% instance là incorrect, ít hơn correct và no_mask. Em đã ghi nhận trong README và đề xuất 3 hướng khắc phục: weighted sampling, copy-paste augmentation từ MaskedFace-Net, hoặc bổ sung FMLD. Em chưa làm vì scope MVP."

**Q**: *"Em nói 'Planned hand-occlusion subclasses'. Em định label như thế nào?"*

**Khung**:
> "Em sẽ tự thu khoảng 200–400 webcam frame chính em + bạn bè, gồm 6 nhãn: 3 nhãn gốc × 2 trạng thái (có/không che tay). Label bằng tool em đã build (FallGuard adapt cho bbox + state). Để tránh overfitting trên ít data, em dùng tập này chỉ làm test set ban đầu, đo performance gap rồi mới quyết định train trên nó. Đây là cách an toàn để không introduce noise vào model production."

#### Câu hỏi mẫu cho Traffic Project

**Q**: *"Tại sao chọn ByteTrack thay vì SORT/DeepSORT?"*

**Khung**:
> "ByteTrack tận dụng cả low-confidence detection thay vì bỏ chúng đi như SORT, nên handle occlusion/motion blur tốt hơn — phù hợp video traffic Việt Nam có xe máy chen lấn. So với DeepSORT thì ByteTrack không cần ReID network, nhẹ hơn, đủ tốt cho fixed camera."

**Q**: *"Em phát hiện duplicate trong CCPD subset thế nào?"*

**Khung**:
> "Em dùng 2 tầng: (1) MD5 hash để tìm exact duplicate sau resize/recompress; (2) perceptual hash (pHash) Hamming distance < 5 để tìm near-duplicate (ảnh khác chút do nén). Tạo contact sheet 4×4 cho mỗi cluster để human review trước khi xóa — vì pHash false-positive ở plate giống nhau nhưng xe khác."

---

### 3.2. Phần B — Technical Fundamentals (~15 phút)

#### Q. *"Phân biệt classification, detection, segmentation."*
- Classification: 1 nhãn / ảnh.
- Detection: nhiều bbox + class.
- Segmentation: nhãn mỗi pixel (semantic) hoặc instance.
- Cho ví dụ ngắn: "Trong dự án Face Mask của em, classification chỉ trả lời 'có người đeo mask không'; detection trả lời 'những người trong khung này, ai đeo, ai không, ai đeo sai, vị trí từng người'; segmentation sẽ trả về vùng pixel chính xác của khẩu trang — vượt nhu cầu MVP."

#### Q. *"YOLO label format?"*
> "Mỗi dòng = 1 object: `class_id x_center y_center width height`, tất cả normalize 0–1 theo kích thước ảnh. Tọa độ là center của bbox, không phải corner — đây là khác biệt với COCO/VOC."

#### Q. *"Train/val/test ratio em chọn thế nào? Tại sao?"*
> "Em mặc định 70/15/15 hoặc 80/10/10 cho dataset > 5k. Quan trọng hơn ratio là: (1) val/test phải đại diện distribution của production; (2) split deterministic để reproduce; (3) nếu có grouping (cùng person/cùng video), phải split theo group, không phải theo file — tránh leak."

#### Q. *"Class imbalance — em xử lý thế nào?"*
> "Có 4 nhóm cách: (1) data-level: oversample minority, undersample majority, copy-paste augmentation; (2) loss-level: focal loss, weighted CE; (3) sampler: WeightedRandomSampler trong PyTorch; (4) eval-level: report per-class metric, không chỉ overall mAP. Trong dự án Face Mask em ưu tiên (1) và (4) vì còn ở giai đoạn MVP."

#### Q. *"Annotation quality em đo bằng gì?"*
- **Inter-annotator agreement** (IAA): cho 2+ annotator label cùng 1 sample, đo Cohen's kappa hoặc IoU agreement với bbox.
- **Self-consistency**: re-label 5% sample sau 1 tuần, so độ ổn định.
- **Spot check**: senior reviewer kiểm 10–20% random.
- **Schema compliance**: tự động check label format, class hợp lệ, bbox không degenerate.

#### Q. *"Real-world data challenges trong Applaydu — em nghĩ sao?"*
> "Theo em đoán có 5 thách thức chính: (1) **occlusion** — tay trẻ em che một phần đồ chơi khi cầm; (2) **motion blur** — trẻ rung tay khi quay camera; (3) **lighting** — phòng khách/phòng ngủ ánh sáng yếu, hỗn hợp đèn vàng/trắng; (4) **reflection** — đồ chơi nhựa Kinder thường bóng, dễ glare; (5) **scale variance** — trẻ đưa camera quá gần hoặc quá xa. Để dataset robust, mỗi class object cần phủ đầy đủ 5 condition trên + tổ hợp giữa chúng. QA cần flag rõ metadata cho từng ảnh: lighting/blur/occlusion/distance."

> Câu này nếu trả lời tốt, **bạn show được hiểu sâu Applaydu mà không cần được brief**.

#### Q. *"Nếu được giao 1.000 object × 50 ảnh/object, em sẽ tổ chức thư mục thế nào?"*
> Đề xuất:
> ```
> dataset/
>   raw/                  # ảnh thô do nhiếp ảnh / crowdsource
>     <object_id>/<capture_session>/<frame>.jpg
>   processed/<version>/  # YOLO format
>     images/{train,val,test}/
>     labels/{train,val,test}/
>     data.yaml
>   metadata/
>     manifest.csv        # 1 dòng/ảnh: object_id, capture_id, lighting, blur, occlusion, ...
>     annotators.csv      # ai label gì, version nào
>   reports/<version>/
>     audit_report.yaml
>     class_distribution.png
>   CHANGELOG.md          # mỗi version: ai làm gì, thêm/xóa class, fix bug nào
> ```
> Nhấn: **immutable raw** (không bao giờ sửa), **versioned processed** (mỗi đợt fix tạo version mới), **manifest trung tâm** để truy ngược ảnh nào từ đâu.

---

### 3.3. Phần C — Behavioral / "Who you are" (~15 phút)

JD ghi rõ "highly detail-oriented, patient, comfortable with repetitive tasks". Họ sẽ probe.

#### Q. *"Em kể một lần làm 1 việc lặp đi lặp lại trong thời gian dài."*

**Khung (STAR)**:
- **S** (Situation): "Trong dự án Face Mask em phải audit 9.205 ảnh PWMFD..."
- **T** (Task): "...để tìm ảnh có annotation sai class hoặc bbox lệch."
- **A** (Action): "Em viết script auto flag, nhưng vẫn phải mở mắt review từng cluster đáng nghi. Mỗi ngày em set quota 500 ảnh, chia làm 3 session 25-phút (Pomodoro), ghi mỗi quyết định vào log để audit lại được."
- **R** (Result): "Sau 3 ngày, em loại được 184 ảnh có annotation lỗi và document lý do từng ảnh. Nhờ đó mAP train ổn định hơn ~2%."

#### Q. *"Em làm gì khi phát hiện một sai sót trong dataset đã gán nhãn?"*

> "Em không tự sửa luôn. Quy trình em theo: (1) ghi vào issue tracker với evidence (screenshot, file path); (2) check xem sai một-lẻ hay sai có pattern (1 annotator? 1 batch?); (3) báo người phụ trách + đề xuất fix; (4) chờ approve rồi mới sửa, lưu version snapshot trước khi sửa. Vì với dataset, sửa sai bằng cách hấp tấp đôi khi tệ hơn để nguyên."

#### Q. *"Em xử lý conflict thế nào khi không đồng ý với senior?"*

> "Em sẽ trình bày dữ liệu và lý do, không tranh luận trên cảm tính. Ví dụ trong dự án Face Mask, ban đầu em định merge dataset andrewmvd + PWMFD trực tiếp, nhưng sau khi đọc paper FMLD em nhận ra 2 nguồn có distribution khác nhau — em quyết định không merge và document lý do. Nếu là với senior, em sẽ pitch quan điểm 1 lần, nếu họ vẫn quyết hướng khác em theo, miễn là em hiểu lý do và quyết định được note lại."

#### Q. *"Em không hiểu thuật ngữ trong meeting — em làm gì?"*

> "Em sẽ note lại và hỏi sau meeting nếu là chi tiết nhỏ, hỏi luôn nếu là thuật ngữ chính ảnh hưởng đến hiểu task. Em không pretend hiểu rồi đoán — risk làm sai task lớn hơn risk hỏi 1 câu ngu."

#### Q. *"Mục tiêu của em sau 6 tháng intern là gì?"*

> "Ngắn hạn: thành thạo workflow dataset của Gameloft, đóng góp được vào quy trình QA. Trung hạn: hiểu pipeline ML production của team đến mức tự đề xuất cải tiến dataset dựa trên model performance — đúng như JD ghi 'work closely with AI engineers to refine data based on model performance and edge cases'. Dài hạn: convert thành fulltime AI Engineer ở Gameloft."

---

### 3.4. Phần D — Câu hỏi cho Interviewer (~5 phút)

**Bắt buộc** chuẩn bị 3–5 câu, hỏi 2–3 câu tùy thời gian. Hỏi câu show curiosity về dataset workflow:

- "Hiện tại team đang dùng tool annotation gì? (CVAT, Label Studio, custom)?"
- "QA workflow hiện tại có inter-annotator agreement check không, hay chủ yếu spot check?"
- "Khi model fail trên edge case, ai là người quyết định bổ sung data class mới — AI engineer hay data team?"
- "Anh/chị từng gặp issue dataset nào ngớ ngẩn nhưng tốn rất nhiều thời gian để debug không?" (câu này rất human, dễ tạo conversation)
- "Lộ trình từ intern → fulltime ở team này thường mất bao lâu, và tiêu chí review là gì?"

**Tránh** hỏi: "Có WFH không?", "Nghỉ trưa mấy giờ?", "Có thưởng Tết không?" — không sai nhưng không tạo điểm cộng.

---

## 4. Vòng 4: Offer

**Mục đích**: thống nhất số liệu cụ thể, ký HĐ.

### 4.1. Trước khi ký, nên xác nhận

| Mục | Hỏi cụ thể |
|-----|------------|
| **Allowance** | "Mức cuối là [X] triệu gross/tháng cho [Y] ngày/tuần, có đúng không ạ?" |
| **Thời hạn** | Bắt đầu khi nào, kết thúc khi nào, có gia hạn không. |
| **Convert fulltime** | Tiêu chí gì để intern → fulltime, timeline thường thế nào. |
| **Working hours** | Giờ vào/ra, có flexible không. |
| **Trang thiết bị** | Công ty cấp laptop hay tự mang? GPU access ra sao? |
| **Onboarding** | Tuần đầu làm gì, mentor là ai. |
| **Confidentiality** | NDA scope thế nào (đặc biệt nếu bạn đang viết public portfolio/blog). |

### 4.2. Câu hỏi negotiate (nếu cần)

Intern Gameloft có policy lương cố định, **negotiate khó thành công**. Tuy nhiên nếu có lý do thực tế:

- "Em có thể làm full-time 5 ngày/tuần thay vì 4. Mức allowance tương ứng sẽ thế nào ạ?"
- "Em có offer khác cao hơn nhưng ưu tiên Gameloft. Anh/chị có thể chia sẻ liệu mức [X] có khả thi không?"

**Đừng bluff** — nếu họ hỏi offer nào, phải trả lời được.

### 4.3. Sau khi ký

- Đọc HĐ kỹ phần **IP ownership** (code/research làm trong giờ thuộc về công ty), **non-compete**, **NDA scope**.
- Confirm bằng email lại các thỏa thuận ngoài HĐ (nếu có).
- Hỏi rõ ngày trả lương hàng tháng và cách trả (chuyển khoản, cần cung cấp tài khoản nào).

---

## 5. Cheat Sheet — One-Liners cần thuộc

### Self-pitch 30 giây (English)
> "I'm Bao, a 4th-year IT student at PTIT focused on Computer Vision. Over the past year I built three CV projects centered on dataset preparation and quality assurance — a labeling tool with human-in-the-loop review, a face mask compliance detector trained on a 9k-image YOLO dataset I converted from PASCAL VOC, and a traffic violation pipeline with a CCPD subset and audit artefacts. I'm applying because I learned that dataset quality drives model quality, and I want to do that at production scale on a real product."

### Project one-liners
- **FallGuard**: "End-to-end dataset assistant: video ingestion, auto YOLO-Pose labeling, Streamlit human review with 13 keypoints, QA reports, versioned export — processed 5 videos to 656 reviewed frames."
- **Face Mask**: "Converted PWMFD from VOC to YOLO (9,205 images, 18,528 instances), trained YOLOv8n on Vast.ai to mAP@50 0.970, planned hand-occlusion subclass extension for real-world deployment."
- **Traffic**: "Fixed-camera pipeline with ByteTrack, traffic-light state, plate OCR, plus a curated CCPD subset of 8,598 images with audit artefacts for missing/blur/duplicate flagging."

### Key numbers cần thuộc lòng
- Face Mask: 9,205 ảnh / 18,528 instances / mAP@50 0.970 / mAP@50:95 0.749
- Traffic: 8,598 ảnh CCPD subset
- FallGuard: 5 videos / 656 frames / 13 keypoints

### 3 trigger phrases recruiter thích nghe
- "I check edge cases before I trust a number."
- "I prefer documenting decisions over fixing things silently."
- "I know dataset bugs are 10× more expensive to fix later than at ingestion."

---

## 6. 24h trước phỏng vấn — Checklist

- [ ] Re-read JD 1 lần, gạch chân từ khóa.
- [ ] Re-read CV của mình, đảm bảo nói được mọi bullet.
- [ ] Luyện self-pitch 30s (Việt + Anh) 3 lần trước gương.
- [ ] Test camera + mic + đường truyền (nếu online).
- [ ] Chuẩn bị 3 câu hỏi cho interviewer.
- [ ] Mở sẵn `audit_dataset.py` và `convert_dataset.py` trên màn hình thứ 2 (nếu online) để tham chiếu nếu được phép.
- [ ] Ngủ đủ. Đi đúng giờ (sớm 10 phút).
- [ ] Mang theo: 2 bản CV in (nếu onsite), notepad + bút, ID card.

---

## 7. Sau phỏng vấn

- Trong vòng **24 giờ**: gửi email cảm ơn ngắn cho người phỏng vấn (qua HR).

```text
Subject: Thank you — AI Engineer Intern Interview — Nguyen Khac Bao

Dear [Name],

Thank you for taking the time to speak with me today about the AI Engineer
Intern role. I especially enjoyed the discussion about [specific topic
discussed, e.g. dataset versioning workflow / handling occlusion edge cases].

If you need any further information from my side, please let me know.
I look forward to hearing from you.

Best regards,
Nguyen Khac Bao
```

- Self-debrief: ghi lại 3 câu trả lời tốt, 3 câu cần cải thiện. Dùng cho lần sau (nếu fail) hoặc lần convert fulltime.

---

**Chốt**: vị trí này về **dataset hygiene + attention to detail**, không phải về thuật toán fancy. Cứ trả lời thật, show được tư duy có hệ thống và tinh thần document, bạn đã hơn 70% ứng viên khác rồi. Good luck!
