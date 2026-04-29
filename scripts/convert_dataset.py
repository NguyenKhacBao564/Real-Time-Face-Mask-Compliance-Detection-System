import argparse
import hashlib
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


CLASS_NAMES = {
    0: "correct_mask",
    1: "incorrect_mask",
    2: "no_mask",
}

CLASS_TO_ID = {value: key for key, value in CLASS_NAMES.items()}

CLASS_MAPPING = {
    "face_with_mask": "correct_mask",
    "with_mask": "correct_mask",
    "mask": "correct_mask",
    "masked_face": "correct_mask",
    "face_mask": "correct_mask",
    "face_without_mask": "no_mask",
    "without_mask": "no_mask",
    "no-mask": "no_mask",
    "unmasked_face": "no_mask",
    "face": "no_mask",
    "face_with_mask_incorrect": "incorrect_mask",
    "incorrect_mask": "incorrect_mask",
    "mask_worn_incorrectly": "incorrect_mask",
    "mask_weared_incorrect": "incorrect_mask",
    "incorrectly_masked_face": "incorrect_mask",
    "nose": "incorrect_mask",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(label: str) -> str | None:
    normalized = label.strip().lower().replace(" ", "_")
    return CLASS_MAPPING.get(normalized)


def infer_split(path: Path, rng: random.Random, val_ratio: float, test_ratio: float) -> str:
    parts = {part.lower() for part in path.parts}
    if {"train", "training"} & parts or any("train" in part for part in parts):
        return "train"
    if {"val", "valid", "validation"} & parts or any(
        "val" in part or "valid" in part for part in parts
    ):
        return "val"
    if {"test", "testing"} & parts or any("test" in part for part in parts):
        return "test"

    value = rng.random()
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def build_image_index(source: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in source.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        index.setdefault(image_path.name, image_path)
        index.setdefault(image_path.stem, image_path)
    return index


def find_image(
    source: Path,
    xml_path: Path,
    filename: str,
    image_index: dict[str, Path],
) -> Path | None:
    candidates = [
        xml_path.parent / filename,
        xml_path.parent.parent / filename,
        source / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    stem = Path(filename).stem
    return image_index.get(xml_path.stem) or image_index.get(filename) or image_index.get(stem)


def parse_voc_xml(
    xml_path: Path,
    source: Path,
    image_index: dict[str, Path],
) -> tuple[str, int, int, list[tuple[int, float, float, float, float]]] | None:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename") or f"{xml_path.stem}.jpg"

    size = root.find("size")
    if size is None:
        return None
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    if width <= 0 or height <= 0:
        return None

    labels: list[tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        source_label = obj.findtext("name", "")
        project_label = normalize_label(source_label)
        if project_label is None:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = max(0.0, float(bbox.findtext("xmin", "0")))
        ymin = max(0.0, float(bbox.findtext("ymin", "0")))
        xmax = min(float(width), float(bbox.findtext("xmax", "0")))
        ymax = min(float(height), float(bbox.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        labels.append((CLASS_TO_ID[project_label], x_center, y_center, box_width, box_height))

    image_path = find_image(source, xml_path, filename, image_index)
    if image_path is None:
        return None
    return str(image_path), width, height, labels


def stable_name(xml_path: Path, image_path: Path) -> str:
    digest = hashlib.sha1(str(xml_path).encode("utf-8")).hexdigest()[:10]
    return f"{image_path.stem}_{digest}"


def write_data_yaml(target: Path) -> None:
    data = {
        "path": str(target),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    (target / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def convert_voc_dataset(source: Path, target: Path, val_ratio: float, test_ratio: float, seed: int) -> dict:
    rng = random.Random(seed)
    xml_paths = sorted(source.rglob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No PASCAL VOC XML files found under {source}")
    image_index = build_image_index(source)

    stats = {
        "xml_files": len(xml_paths),
        "converted_images": 0,
        "skipped_xml": 0,
        "empty_label_files": 0,
        "split_images": {"train": 0, "val": 0, "test": 0},
        "class_counts": {name: 0 for name in CLASS_NAMES.values()},
    }

    for split in ("train", "val", "test"):
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)

    for xml_path in xml_paths:
        parsed = parse_voc_xml(xml_path, source, image_index)
        if parsed is None:
            stats["skipped_xml"] += 1
            continue

        image_path = Path(parsed[0])
        labels = parsed[3]
        split = infer_split(xml_path.relative_to(source), rng, val_ratio, test_ratio)
        name = stable_name(xml_path.relative_to(source), image_path)
        target_image = target / "images" / split / f"{name}{image_path.suffix.lower()}"
        target_label = target / "labels" / split / f"{name}.txt"

        shutil.copy2(image_path, target_image)
        lines = []
        for class_id, x_center, y_center, box_width, box_height in labels:
            class_name = CLASS_NAMES[class_id]
            stats["class_counts"][class_name] += 1
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if not lines:
            stats["empty_label_files"] += 1
        stats["converted_images"] += 1
        stats["split_images"][split] += 1

    write_data_yaml(target)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Raw PASCAL VOC dataset directory")
    parser.add_argument("--target", default="data/mask_detection", help="Processed YOLO dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    stats = convert_voc_dataset(source, target, args.val_ratio, args.test_ratio, args.seed)
    print(yaml.safe_dump(stats, sort_keys=False))


if __name__ == "__main__":
    main()
