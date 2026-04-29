# Next Steps

Do these in order. The goal is to get one end-to-end baseline before adding complexity.

## Step 1: Prepare Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Get A Small Smoke-Test Dataset

Use `andrewmvd/face-mask-detection` first only to verify the pipeline.

```bash
python scripts/download_dataset.py --dataset andrewmvd
```

Inspect the downloaded path from `data/raw/download_manifest.json`, then convert:

```bash
python scripts/convert_dataset.py \
  --source <downloaded_dataset_path> \
  --target data/mask_detection
```

Audit:

```bash
python scripts/audit_dataset.py --dataset data/mask_detection
```

## Step 3: Run A Local Smoke Train

Use a very small epoch count just to verify training works:

```bash
bash scripts/train_smoke.sh
```

Do not judge model quality from this run.

If Ultralytics saves runs into an old project, check:

```bash
yolo settings
```

The local smoke script uses `project="$(pwd)/runs/train"` to force outputs into this repository.

## Step 4: Prepare PWMFD For The First Real Model

Download PWMFD train/validation archives from the official repository links, place them under:

```text
data/raw/pwmfd/
```

Then convert and audit:

```bash
python scripts/convert_dataset.py --source data/raw/pwmfd --target data/mask_detection
python scripts/audit_dataset.py --dataset data/mask_detection
```

If `incorrect_mask` is too small, continue anyway but document it in the report.

## Step 5: Train On Vast.ai

Upload the repo and dataset archive, or let the Vast.ai instance download the archive from your cloud storage.

```bash
bash scripts/train_vast.sh
```

Before stopping the instance, download:

```text
runs/train/<run>/weights/best.pt
runs/train/<run>/results.csv
runs/train/<run>/confusion_matrix.png
runs/train/<run>/PR_curve.png
```

## Step 6: Test The Demo App

Put the trained weight here:

```text
models/best.pt
```

Run:

```bash
bash scripts/run_server.sh
```

Open:

```text
http://localhost:8000
```

## Step 7: Write Results Honestly

Update README with:

```text
hardware specs
dataset version
mAP@50 and mAP@50:95
per-class precision/recall
local FPS and latency
failure cases, especially incorrect_mask
```
