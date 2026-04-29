#!/usr/bin/env bash
set -euo pipefail

yolo detect train \
  model=yolov8n.pt \
  data=data/mask_detection/data.yaml \
  epochs="${EPOCHS:-3}" \
  imgsz="${IMGSZ:-512}" \
  batch="${BATCH:-8}" \
  project="$(pwd)/runs/train" \
  name="${NAME:-smoke_test}" \
  device="${DEVICE:-cpu}"
