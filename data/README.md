# Data Directory

This directory stores local datasets only. Raw and processed image data should not be committed.

Expected local layout after downloading and conversion:

```text
data/
├── raw/
└── mask_detection/
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

Use `scripts/download_dataset.py`, `scripts/convert_dataset.py`, and `scripts/audit_dataset.py` to create this structure.

