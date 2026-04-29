import argparse
from pathlib import Path


CLASS_MAPPING = {
    "with_mask": "correct_mask",
    "mask": "correct_mask",
    "without_mask": "no_mask",
    "no-mask": "no_mask",
    "mask_worn_incorrectly": "incorrect_mask",
    "mask_weared_incorrect": "incorrect_mask",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Raw dataset directory")
    parser.add_argument("--target", default="data/mask_detection", help="Processed YOLO dataset directory")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source}")
    print(f"Target: {target}")
    print("Converter scaffold created. Implement dataset-specific conversion after inspecting the downloaded layout.")
    print("Class mapping:", CLASS_MAPPING)


if __name__ == "__main__":
    main()

