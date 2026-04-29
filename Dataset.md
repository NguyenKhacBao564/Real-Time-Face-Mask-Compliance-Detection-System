# Dataset Plan

This project needs an object detection dataset, not only a face classification dataset. The training target is one YOLO detector with three classes:

```yaml
names:
  0: correct_mask
  1: incorrect_mask
  2: no_mask
```

## Recommendation

Use this staged approach:

```text
V1 baseline:
Use a clean YOLO-format dataset to verify the full training pipeline.

V2 3-class model:
Use a dataset with incorrect_mask annotations and convert it to YOLO if needed.

V3 real-world test:
Collect 100-300 webcam frames for evaluation only, not for training at first.
```

Do not merge a large 2-class dataset and a small 3-class dataset blindly. That can make `incorrect_mask` underrepresented and unreliable.

## Dataset Comparison

| Dataset | Format | Classes | Best role | Main concern |
| --- | --- | --- | --- | --- |
| `parot99/face-mask-detection-yolo-darknet-format` | YOLO/Darknet | `mask`, `no-mask` | Primary 2-class baseline | No `incorrect_mask` class |
| `andrewmvd/face-mask-detection` | PASCAL VOC | `with_mask`, `without_mask`, `mask_worn_incorrectly` | 3-class fine-tune / auxiliary dataset | Small dataset, needs VOC to YOLO conversion |
| `aditya276/face-mask-dataset-yolo-format` | YOLO | mostly `with_mask`, `without_mask` | Fast backup baseline | Less useful for `incorrect_mask` |
| `cabani/MaskedFace-Net` | face-level images | correctly masked, incorrectly masked | Future classification/pretraining experiment | Very large, not YOLO full-scene detection data |

## Is MaskedFace-Net Actually Bad?

No. `MaskedFace-Net` is not a bad dataset. It is just not the right first dataset for this MVP.

The project page describes it as a dataset of human faces with correctly and incorrectly worn masks based on FFHQ. It has 133,783 images: 67,049 correctly masked face images and 66,734 incorrectly masked face images. The downloads are large: about 19 GB for CMFD and 19 GB for IMFD.

Why it is not ideal for this MVP:

1. It is face-level data, not a ready YOLO full-scene object detection dataset.
2. It does not directly provide the `no_mask` class needed for the final 3-class detector.
3. It is too large for quick iteration on Vast.ai unless there is a strong reason.
4. It has a non-commercial ShareAlike-style license, so public portfolio and reuse need careful attribution and license handling.
5. Using it would likely push the project toward a classifier pipeline instead of the simpler YOLO detector MVP.

When it could be useful:

```text
Future work:
- train a correct_mask vs incorrect_mask classifier
- pretrain a face crop classifier
- create a two-stage pipeline: face detector -> mask correctness classifier
- build a research comparison, not the MVP
```

For the current project, use `MaskedFace-Net` only as a documented future improvement.

## Class Mapping

Normalize all labels to this schema:

| Source label | Project label |
| --- | --- |
| `with_mask` | `correct_mask` |
| `mask` | `correct_mask` |
| `without_mask` | `no_mask` |
| `no-mask` | `no_mask` |
| `mask_worn_incorrectly` | `incorrect_mask` |
| `mask_weared_incorrect` | `incorrect_mask` |

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

- [MaskedFace-Net GitHub](https://github.com/cabani/MaskedFace-Net)
- [Kaggle: parot99 Face Mask Detection YOLO Darknet Format](https://www.kaggle.com/datasets/parot99/face-mask-detection-yolo-darknet-format)
- [Kaggle: andrewmvd Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [Kaggle: aditya276 Face Mask Dataset YOLO Format](https://www.kaggle.com/datasets/aditya276/face-mask-dataset-yolo-format)
