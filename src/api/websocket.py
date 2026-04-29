import base64
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.inference.predictor import MaskDetector
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


@router.websocket("/api/v1/ws/detect")
async def detect_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    frame_id = 0

    try:
        while True:
            frame_id += 1
            message = await websocket.receive()
            started = time.perf_counter()

            if "bytes" in message and message["bytes"] is not None:
                image_bytes = message["bytes"]
            elif "text" in message and message["text"] is not None:
                text = message["text"]
                image_bytes = base64.b64decode(text.split(",", 1)[-1])
            else:
                await websocket.send_json({"error": "empty frame", "frame_id": frame_id})
                continue

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                tmp.write(image_bytes)
                tmp.flush()
                try:
                    result = get_predictor().predict_image(tmp.name)
                except FileNotFoundError as exc:
                    await websocket.send_json({"error": str(exc), "frame_id": frame_id})
                    continue

            latency_ms = (time.perf_counter() - started) * 1000
            result["frame_id"] = frame_id
            result["latency_ms"] = round(latency_ms, 2)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return

