import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, Response
from prometheus_client import generate_latest

from src.api.predictor_provider import get_predictor
from src.inference.predictor import PredictorError
from src.utils.dataset_capture import save_real_world_sample
from src.utils.events import emit_events_from_result
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger("api")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
def get_metrics():
    return Response(content=generate_latest(), media_type="text/plain")


@router.post("/api/v1/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    client_id: Optional[str] = Query(None, description="Optional client/session id"),
) -> dict:
    request_started = time.perf_counter()
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    image_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        try:
            result = get_predictor().predict_image(tmp.name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except PredictorError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    events = emit_events_from_result(
        result,
        source="image",
        image_bytes=image_bytes,
        client_id=client_id,
    )

    request_latency_ms = round((time.perf_counter() - request_started) * 1000, 2)
    logger.info(
        "rest_predict source=image client_id=%s inference_ms=%s request_ms=%s "
        "detections=%d violations_emitted=%d",
        client_id or "-",
        result.get("latency_ms"),
        request_latency_ms,
        len(result.get("detections") or []),
        len(events),
    )
    return result


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
