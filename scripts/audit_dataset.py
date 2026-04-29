import argparse
from collections import Counter
from pathlib import Path

import yaml


def audit_yolo_dataset(dataset_dir: Path) -> dict:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names", {})
    report: dict[str, dict] = {}

    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_paths = list(image_dir.glob("*")) if image_dir.exists() else []
        label_paths = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        class_counts: Counter[str] = Counter()
        malformed_lines = 0

        for label_path in label_paths:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    malformed_lines += 1
                    continue
                class_name = names.get(int(parts[0]), parts[0])
                class_counts[str(class_name)] += 1

        report[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "class_counts": dict(class_counts),
            "malformed_lines": malformed_lines,
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mask_detection")
    args = parser.parse_args()

    report = audit_yolo_dataset(Path(args.dataset))
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()

