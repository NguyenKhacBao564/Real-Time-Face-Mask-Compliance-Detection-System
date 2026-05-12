import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.predictor_provider import get_predictor
from src.inference.predictor import PredictorError
from src.utils.dataset_capture import save_real_world_sample

router = APIRouter()


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


@router.post("/api/v1/dataset/capture")
async def capture_dataset_sample(
    file: UploadFile = File(...),
    label: str = Form(...),
    subtype: str = Form(""),
    lighting: str = Form("normal"),
    occlusion: str = Form("none"),
    blur: str = Form("none"),
    reflection: str = Form("none"),
    notes: str = Form(""),
) -> dict:
    image_bytes = await file.read()
    try:
        return save_real_world_sample(
            image_bytes=image_bytes,
            label=label,
            subtype=subtype,
            lighting=lighting,
            occlusion=occlusion,
            blur=blur,
            reflection=reflection,
            notes=notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
