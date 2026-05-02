# Model Report: mask_yolov8n_pwmfd_baseline

Training artifact:

```text
mask_yolov8n_pwmfd_baseline/
```

Weights:

```text
mask_yolov8n_pwmfd_baseline/weights/best.pt
mask_yolov8n_pwmfd_baseline/weights/best.onnx
```

The best weights have also been copied to:

```text
models/best.pt
models/best.onnx
```

## Training Setup

```text
Model: YOLOv8n
Dataset: PWMFD converted to YOLO
Epochs: 50
Image size: 640
Batch: 16
Device: RTX 3090 24GB on Vast.ai
AMP: true
```

## Overall Metrics

Best epoch by mAP@50:

| Epoch | Precision | Recall | mAP@50 | mAP@50:95 |
| ---: | ---: | ---: | ---: | ---: |
| 45 | 0.940 | 0.940 | 0.969 | 0.742 |

Last epoch:

| Epoch | Precision | Recall | mAP@50 | mAP@50:95 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 0.944 | 0.945 | 0.968 | 0.741 |

## Per-Class Validation

Local validation on the PWMFD validation split:

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1,820 | 1,826 | 0.947 | 0.938 | 0.970 | 0.749 |
| correct_mask | 986 | 990 | 0.975 | 0.956 | 0.982 | 0.743 |
| incorrect_mask | 46 | 46 | 0.919 | 0.913 | 0.962 | 0.769 |
| no_mask | 790 | 790 | 0.947 | 0.946 | 0.965 | 0.736 |

CPU validation speed on Apple M2 Pro:

```text
0.4 ms preprocess
139.1 ms inference
0.3 ms postprocess
```

## Assessment

This is a strong baseline. The main project risk was `incorrect_mask`, and this run detects it well on PWMFD validation data. However, validation has only 46 `incorrect_mask` instances, so a webcam/real-world test set is still required before claiming robust real-world compliance detection.

Next step:

```text
Run the local FastAPI/webcam demo with models/best.pt, then collect 100-200 real webcam frames for an external test set.
```
