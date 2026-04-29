import argparse
from pathlib import Path

from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)

    from ultralytics import YOLO

    model = YOLO(config["model"])
    project = Path(config["project"])
    if not project.is_absolute():
        project = Path.cwd() / project

    model.train(
        data=config["data"],
        epochs=int(config["epochs"]),
        imgsz=int(config["image_size"]),
        batch=int(config["batch"]),
        device=config["device"],
        project=str(project),
        name=config["name"],
        patience=int(config.get("patience", 15)),
    )


if __name__ == "__main__":
    main()
