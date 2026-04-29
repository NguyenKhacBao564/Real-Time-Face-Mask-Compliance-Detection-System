# Training on Vast.ai

Use Vast.ai for GPU training and evaluation, not as the source of truth for the dataset. Vast.ai instances are often temporary, so the workflow should be reproducible from the Git repo plus external dataset sources.

## Recommended Data Strategy

Do not push datasets to GitHub.

Use this strategy instead:

```text
Git repo:
- code
- configs
- dataset download scripts
- dataset conversion scripts
- data.yaml template
- documentation

External storage:
- Kaggle datasets
- Hugging Face Dataset repo
- Google Drive / S3 / R2 bucket
- GitHub Release only for small artifacts, not raw datasets

Vast.ai instance:
- downloads raw data into data/raw/
- converts processed data into data/mask_detection/
- trains from data/mask_detection/data.yaml
```

For the current MVP, the best workflow is:

```text
1. Download or upload raw dataset sources to data/raw/ on Vast.ai.
2. Convert/audit them into YOLO format on Vast.ai.
3. Train the model.
4. Download only results, metrics, and final weights.
5. Optionally upload a processed dataset archive to external storage for future repeat runs.
```

## Should Vast.ai Download the Dataset Automatically?

Yes, when the dataset has a stable direct download or Kaggle source. It is better than manually uploading data every time.

For PWMFD, the repository points to external OneDrive/Baidu downloads. If direct download automation is unreliable, download the train/validation archives locally once, upload them to private cloud storage, then let Vast.ai download that archive.

For FMLD, the annotations are on GitHub, but the underlying MAFA/WIDER Face images must be downloaded separately. Treat FMLD as Stage 3, not the first Vast.ai run.

Recommended:

```text
scripts/download_dataset.py
scripts/convert_dataset.py
scripts/audit_dataset.py
scripts/train_vast.sh
```

For very large datasets like `MaskedFace-Net`, do not auto-download it in the MVP. A 38 GB dataset makes iteration slow and expensive, and it still does not directly solve YOLO full-scene detection.

## Remote Directory Layout

On Vast.ai:

```text
/workspace/realtime-face-mask-detection/
├── data/
│   ├── raw/
│   └── mask_detection/
├── models/
├── outputs/
└── runs/
```

`data/raw/` stores downloaded source datasets.  
`data/mask_detection/` stores the final YOLO-format dataset used for training.

## Kaggle Setup

On your local machine, create a Kaggle API token from Kaggle account settings. On Vast.ai, put `kaggle.json` here:

```text
~/.kaggle/kaggle.json
```

Then set permission:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Install dependencies:

```bash
pip install kaggle kagglehub ultralytics opencv-python pandas matplotlib tqdm pyyaml
```

Example dataset download:

```python
import kagglehub

path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
print("Dataset path:", path)
```

Generic VOC-to-YOLO conversion after downloading a PASCAL VOC style dataset:

```bash
python scripts/convert_dataset.py \
  --source data/raw/pwmfd \
  --target data/mask_detection
```

Then audit:

```bash
python scripts/audit_dataset.py --dataset data/mask_detection
```

## Training Command

Baseline:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/mask_detection/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=mask_yolov8n_baseline
```

Recommended first real run name:

```text
mask_yolov8n_pwmfd_baseline
```

If GPU memory is limited:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/mask_detection/data.yaml \
  epochs=50 \
  imgsz=512 \
  batch=8 \
  device=0 \
  project=runs/train \
  name=mask_yolov8n_baseline_512
```

## Evaluation

```bash
yolo detect val \
  model=runs/train/mask_yolov8n_baseline/weights/best.pt \
  data=data/mask_detection/data.yaml \
  imgsz=640 \
  device=0
```

Save:

```text
results.csv
confusion_matrix.png
PR_curve.png
F1_curve.png
sample prediction images
```

## Export

Export ONNX after the `.pt` model is good enough:

```bash
yolo export \
  model=runs/train/mask_yolov8n_baseline/weights/best.pt \
  format=onnx \
  imgsz=640
```

Store large model files outside Git:

```text
models/best.pt
models/best.onnx
```

Then document the download location in `models/README.md`.

## Persistent Storage Advice

If you will train multiple times:

1. Keep a compressed processed dataset archive outside the repo.
2. Upload it to Hugging Face Datasets, Google Drive, S3, or Cloudflare R2.
3. Make the Vast.ai setup script download that archive directly.

This avoids repeating the full Kaggle download and conversion every time.

Suggested archive:

```text
mask_detection_yolo_v1.tar.gz
```

Inside:

```text
images/train/
images/val/
images/test/
labels/train/
labels/val/
labels/test/
data.yaml
dataset_report.md
```

## Minimum Vast.ai Checklist

Before leaving the instance, download or upload:

```text
runs/train/<run_name>/weights/best.pt
runs/train/<run_name>/results.csv
runs/train/<run_name>/confusion_matrix.png
runs/train/<run_name>/PR_curve.png
runs/train/<run_name>/F1_curve.png
runs/train/<run_name>/val_batch*_pred.jpg
```

Do not rely on the instance staying alive.
