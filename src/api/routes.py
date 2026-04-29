import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.inference.predictor import MaskDetector, PredictorError
from src.utils.config import load_app_config

router = APIRouter()

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


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/predict/image")
async def predict_image(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    image_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        try:
            return get_predictor().predict_image(tmp.name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except PredictorError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

