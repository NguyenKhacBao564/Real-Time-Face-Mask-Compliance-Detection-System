import argparse
import json
from pathlib import Path


DATASETS = {
    "parot99": "parot99/face-mask-detection-yolo-darknet-format",
    "andrewmvd": "andrewmvd/face-mask-detection",
    "aditya276": "aditya276/face-mask-dataset-yolo-format",
}

MANUAL_DATASETS = {
    "pwmfd": "https://github.com/ethancvaa/Properly-Wearing-Masked-Detect-Dataset",
    "fmld": "https://github.com/borutb-fri/FMLD",
    "aizoo": "https://github.com/AIZOOTech/FaceMaskDetection",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted({*DATASETS.keys(), *MANUAL_DATASETS.keys()}), required=True)
    parser.add_argument("--output", default="data/raw/download_manifest.json")
    args = parser.parse_args()

    if args.dataset in MANUAL_DATASETS:
        manifest = {
            "name": args.dataset,
            "source": MANUAL_DATASETS[args.dataset],
            "path": None,
            "note": "Manual download is required; place extracted files under data/raw/<dataset_name>/.",
        }
    else:
        import kagglehub

        dataset_id = DATASETS[args.dataset]
        path = kagglehub.dataset_download(dataset_id)
        manifest = {"name": args.dataset, "dataset_id": dataset_id, "path": path}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
