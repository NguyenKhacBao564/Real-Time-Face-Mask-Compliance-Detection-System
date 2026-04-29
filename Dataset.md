# Dataset Plan

This project needs an object detection dataset, not only a face classification dataset. The target is one YOLO detector with three classes:

```yaml
names:
  0: correct_mask
  1: incorrect_mask
  2: no_mask
```

## Recommended Strategy

Use this staged dataset plan:

```text
Stage 1: smoke test
Dataset: andrewmvd/face-mask-detection
Goal: verify download -> convert -> audit -> train -> predict works.
Metric expectation: not important.

Stage 2: first real baseline
Dataset: PWMFD
Goal: train the first presentable 3-class YOLOv8n model.
Metric expectation: present mAP and per-class precision/recall honestly.

Stage 3: improve incorrect_mask
Dataset: PWMFD + FMLD, merged only after audit.
Goal: improve recall and robustness for incorrectly worn masks.

Stage 4: real-world test
Dataset: 100-200 self-collected webcam frames.
Goal: evaluate domain gap only; do not train on these frames first.
```

The main risk is `incorrect_mask`. It is a minority class in many public datasets, so the project must report per-class metrics instead of hiding behind overall mAP.

## Dataset Comparison

| Dataset | Format | Classes | Best role | Main concern |
| --- | --- | --- | --- | --- |
| PWMFD | PASCAL VOC XML | `face_with_mask`, `face_with_mask_incorrect`, `face_without_mask` | First real 3-class baseline | Incorrect-mask class is still much smaller than other classes |
| FMLD | PASCAL VOC XML annotations | `masked_face`, `incorrectly_masked_face`, `unmasked_face` | Improve `incorrect_mask` and real-world robustness | Requires MAFA/WIDER source images and more preprocessing |
| `andrewmvd/face-mask-detection` | PASCAL VOC XML | `with_mask`, `without_mask`, `mask_worn_incorrectly` | Fast smoke test | Too small for final model |
| AIZOO FaceMaskDetection | custom/source repo | `face`, `face_mask` | Optional 2-class reference baseline | No `incorrect_mask` class |
| `parot99/face-mask-detection-yolo-darknet-format` | YOLO/Darknet | `mask`, `no-mask` | Backup 2-class baseline | No `incorrect_mask` class |
| `aditya276/face-mask-dataset-yolo-format` | YOLO | mostly `with_mask`, `without_mask` | Fast backup baseline | Less useful for compliance |
| `cabani/MaskedFace-Net` | face-level images | correctly masked, incorrectly masked | Future classifier/pretraining experiment | Very large, synthetic face crops, no YOLO scene bboxes, no `no_mask` |

Notes from source checks:

- PWMFD reports 9,205 images and 18,532 labeled instances, with train/validation splits and three mask-wearing states.
- FMLD reports 63,072 face images, annotations derived from MAFA and WIDER Face, three mask states, and PASCAL VOC XML annotation files.
- AIZOO reports 7,971 images from WIDER Face and MAFA for face mask detection, but it is a 2-class direction.

## Is MaskedFace-Net Actually Bad?

No. `MaskedFace-Net` is not a bad dataset. It is just not the right first dataset for this MVP.

Why it is not ideal for the current YOLO detector:

1. It is face-level crop data, not ready full-scene YOLO detection data.
2. It does not directly provide the `no_mask` class.
3. It is synthetic, so the visual domain can differ from webcam footage.
4. It is about 38 GB total, which slows Vast.ai iteration.
5. Its CC BY-NC-SA style license needs care for public portfolio and model reuse.

When it could be useful:

```text
Future work:
- train a correct_mask vs incorrect_mask classifier
- pretrain a two-stage face-crop classifier
- experiment with copy-paste augmentation for rare incorrect_mask examples
```

For the current project, keep it as future work.

## Class Mapping

Normalize source labels to the project schema:

| Source label | Project label |
| --- | --- |
| `face_with_mask` | `correct_mask` |
| `with_mask` | `correct_mask` |
| `mask` | `correct_mask` |
| `masked_face` | `correct_mask` |
| `face_without_mask` | `no_mask` |
| `without_mask` | `no_mask` |
| `no-mask` | `no_mask` |
| `unmasked_face` | `no_mask` |
| `face_with_mask_incorrect` | `incorrect_mask` |
| `mask_worn_incorrectly` | `incorrect_mask` |
| `mask_weared_incorrect` | `incorrect_mask` |
| `incorrectly_masked_face` | `incorrect_mask` |

## YOLO Dataset Layout

Final processed dataset should look like this:

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

One YOLO label line:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to `0.0-1.0`.

## Dataset Audit Checklist

Before training, create a dataset report with:

```text
number of images per split
number of labels per class
empty label files
missing label files
broken images
duplicate images
bbox coordinates outside 0-1
very tiny or huge boxes
sample visualization grid
license notes
```

Minimum expected artifacts:

```text
outputs/dataset_report.md
outputs/class_distribution.png
outputs/sample_labels_grid.jpg
```

## What Goes Into Git?

Commit:

```text
dataset scripts
conversion scripts
small config files
data/README.md
data.yaml template
dataset audit report
```

Do not commit:

```text
raw downloaded datasets
processed image folders
training runs
model weights
large generated outputs
```

Use `.gitignore` for:

```text
data/raw/
data/mask_detection/
runs/
outputs/
models/*.pt
models/*.onnx
```

## References

- [PWMFD: Properly-Wearing-Masked Detect Dataset](https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset)
- [FMLD: Face Mask Label Dataset](https://github.com/borutb-fri/FMLD)
- [FMLD paper: How to Correctly Detect Face-Masks for COVID-19](https://www.mdpi.com/2076-3417/11/5/2070)
- [AIZOO FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection)
- [MaskedFace-Net GitHub](https://github.com/cabani/MaskedFace-Net)
- [Kaggle: andrewmvd Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
