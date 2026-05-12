import os

from src.inference.predictor import MaskDetector
from src.utils.config import load_app_config

_predictor: MaskDetector | None = None


def get_predictor() -> MaskDetector:
    global _predictor
    if _predictor is None:
        config = load_app_config()
        model_path = os.getenv("MODEL_PATH", config["model"]["path"])
        _predictor = MaskDetector(
            model_path=model_path,
            image_size=int(config["model"]["image_size"]),
            conf=float(config["model"]["confidence_threshold"]),
            iou=float(config["model"]["iou_threshold"]),
        )
    return _predictor


def preload_predictor() -> None:
    get_predictor().load()

