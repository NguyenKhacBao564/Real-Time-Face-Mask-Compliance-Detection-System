import argparse
import json
from pathlib import Path

import kagglehub


DATASETS = {
    "parot99": "parot99/face-mask-detection-yolo-darknet-format",
    "andrewmvd": "andrewmvd/face-mask-detection",
    "aditya276": "aditya276/face-mask-dataset-yolo-format",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--output", default="data/raw/download_manifest.json")
    args = parser.parse_args()

    dataset_id = DATASETS[args.dataset]
    path = kagglehub.dataset_download(dataset_id)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"name": args.dataset, "dataset_id": dataset_id, "path": path}
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

