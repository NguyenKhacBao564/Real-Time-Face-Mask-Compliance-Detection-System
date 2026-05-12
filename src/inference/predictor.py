import time
from pathlib import Path
from typing import Any

from src.inference.postprocess import count_detections, parse_ultralytics_result


class PredictorError(RuntimeError):
    pass


class MaskDetector:
    def __init__(
        self,
        model_path: str = "models/best.pt",
        image_size: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.conf = conf
        self.iou = iou
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. Train a model or place weights in models/best.pt."
            )

        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise PredictorError("ultralytics is not installed") from exc

            self._model = YOLO(str(self.model_path))
        return self._model

    def load(self) -> None:
        self._load_model()

    def predict_image(self, image_path: str) -> dict:
        started = time.perf_counter()
        model = self._load_model()

        results = model.predict(
            source=image_path,
            imgsz=self.image_size,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        if not results:
            detections: list[dict] = []
        else:
            detections = parse_ultralytics_result(results[0])

        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "detections": detections,
            "counts": count_detections(detections),
            "latency_ms": round(latency_ms, 2),
        }
