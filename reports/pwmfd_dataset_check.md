# PWMFD Dataset Check

Source path:

```text
data/raw/pwmfd/
```

Raw structure:

```text
data/raw/pwmfd/PWMFD_Train/Annotations
data/raw/pwmfd/PWMFD_Train/Pictures
data/raw/pwmfd/PWMFD_Val/Annotations
data/raw/pwmfd/PWMFD_Val/Pictures
```

Raw file count:

```text
XML files: 9,205
image files: 9,205
```

Raw labels found:

| Raw label | Count | Project label |
| --- | ---: | --- |
| `without_mask` | 9,680 | `no_mask` |
| `with_mask` | 6,702 | `correct_mask` |
| `face_mask` | 993 | `correct_mask` |
| `face` | 791 | `no_mask` |
| `incorrect_mask` | 320 | `incorrect_mask` |
| `nose` | 46 | `incorrect_mask` |

Conversion result:

```text
xml_files: 9205
converted_images: 9205
skipped_xml: 0
empty_label_files: 5
split_images:
  train: 7385
  val: 1820
  test: 0
class_counts:
  correct_mask: 7692
  incorrect_mask: 366
  no_mask: 10470
```

YOLO audit:

| Split | Images | correct_mask | incorrect_mask | no_mask | Malformed lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 7,385 | 6,702 | 320 | 9,680 | 0 |
| val | 1,820 | 990 | 46 | 790 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

Notes:

- The dataset has no separate test split; use validation for training feedback and collect a small webcam test set later.
- `incorrect_mask` is still highly underrepresented: 366 of 18,528 labeled instances.
- 5 label files are empty because their XML files have no object or invalid bbox coordinates. This is negligible for training, but keep it documented.
- Converter had to prioritize image lookup by XML stem because some validation XML `filename` fields refer to original source names while the actual image files are numbered.

Next step:

```text
Train YOLOv8n baseline on PWMFD, preferably on Vast.ai GPU.
```
