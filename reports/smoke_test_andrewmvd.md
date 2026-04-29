# Smoke Test: andrewmvd Face Mask Detection

This run verifies that the local pipeline works end to end:

```text
download -> VOC XML conversion -> YOLO audit -> YOLOv8n train -> validation
```

This is not the final model. The dataset is too small and the `incorrect_mask` class is underrepresented.

## Dataset Audit

Converted dataset:

```text
xml_files: 853
converted_images: 853
skipped_xml: 0
empty_label_files: 0
```

Class counts:

| Class | Instances |
| --- | ---: |
| correct_mask | 3,232 |
| incorrect_mask | 123 |
| no_mask | 717 |

Split counts:

| Split | Images | correct_mask | incorrect_mask | no_mask |
| --- | ---: | ---: | ---: | ---: |
| train | 644 | 2,441 | 97 | 503 |
| val | 131 | 497 | 15 | 144 |
| test | 78 | 294 | 11 | 70 |

## Smoke Training

Command:

```bash
yolo detect train model=yolov8n.pt data=data/mask_detection/data.yaml epochs=3 imgsz=512 batch=8 project=runs/train name=smoke_test
```

Hardware:

```text
Apple M2 Pro CPU
```

Final validation metrics after 3 epochs:

| Class | Precision | Recall | mAP@50 | mAP@50:95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.553 | 0.451 | 0.504 | 0.320 |
| correct_mask | 0.752 | 0.872 | 0.886 | 0.590 |
| incorrect_mask | 0.265 | 0.133 | 0.177 | 0.143 |
| no_mask | 0.642 | 0.347 | 0.449 | 0.229 |

Speed:

```text
87.2 ms inference per image on CPU
```

Local benchmark after warmup with `scripts/benchmark.py` on 78 test images:

| Images | Avg latency | P95 latency | FPS |
| ---: | ---: | ---: | ---: |
| 78 | 37.98 ms | 44.24 ms | 26.33 |

## API Smoke Test

The smoke model was copied to `models/best.pt` for local API testing.

```text
GET /health -> 200 OK
POST /api/v1/predict/image -> 200 OK
```

Sample image response contained:

```text
correct_mask: 2
incorrect_mask: 0
no_mask: 1
```

The first request had cold-start latency because the model was loaded on demand. Benchmark numbers above are the better reference for warmed inference.

## Interpretation

The pipeline is working. The weak `incorrect_mask` result is expected because validation has only 15 `incorrect_mask` instances. This confirms that andrewmvd is useful for smoke testing, but not strong enough as the final dataset.

Next dataset should be PWMFD.
