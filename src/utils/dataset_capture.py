import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4


VALID_LABELS = {"correct_mask", "incorrect_mask", "no_mask"}
VALID_LIGHTING = {"normal", "low_light", "backlight", "strong_light"}
VALID_OCCLUSION = {"none", "hand", "arm", "partial_face", "object"}
VALID_BLUR = {"none", "motion_blur", "out_of_focus"}
VALID_REFLECTION = {"none", "mild", "strong"}

DATASET_DIR = Path("outputs/real_world_test")
IMAGE_DIR = DATASET_DIR / "images"
METADATA_PATH = DATASET_DIR / "metadata.csv"

FIELDNAMES = [
    "image_path",
    "timestamp_utc",
    "label",
    "subtype",
    "lighting",
    "occlusion",
    "blur",
    "reflection",
    "source",
    "notes",
]


def validate_choice(value: str, valid_values: set[str], field_name: str) -> None:
    if value not in valid_values:
        raise ValueError(f"Invalid {field_name}: {value}. Expected one of {sorted(valid_values)}")


def save_real_world_sample(
    image_bytes: bytes,
    label: str,
    subtype: str,
    lighting: str,
    occlusion: str,
    blur: str,
    reflection: str,
    notes: str,
) -> dict:
    validate_choice(label, VALID_LABELS, "label")
    validate_choice(lighting, VALID_LIGHTING, "lighting")
    validate_choice(occlusion, VALID_OCCLUSION, "occlusion")
    validate_choice(blur, VALID_BLUR, "blur")
    validate_choice(reflection, VALID_REFLECTION, "reflection")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{timestamp}_{label}_{uuid4().hex[:8]}.jpg"
    image_path = IMAGE_DIR / filename
    image_path.write_bytes(image_bytes)

    row = {
        "image_path": str(image_path),
        "timestamp_utc": timestamp,
        "label": label,
        "subtype": subtype,
        "lighting": lighting,
        "occlusion": occlusion,
        "blur": blur,
        "reflection": reflection,
        "source": "webcam",
        "notes": notes,
    }

    write_header = not METADATA_PATH.exists()
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return {"saved": True, "image_path": str(image_path), "metadata_path": str(METADATA_PATH)}
