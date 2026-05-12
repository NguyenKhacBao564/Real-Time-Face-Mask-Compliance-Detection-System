import tempfile
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.predictor_provider import get_predictor
from src.utils.config import load_app_config
from src.utils.snapshots import has_violation, save_violation_snapshot

router = APIRouter()


@router.websocket("/api/v1/ws/detect")
async def detect_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    frame_id = 0
    config = load_app_config()
    runtime = config.get("runtime", {})
    save_snapshots = bool(runtime.get("save_violation_snapshots", True))
    snapshot_dir = runtime.get("snapshot_dir", "outputs/snapshots")

    try:
        while True:
            frame_id += 1
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            started = time.perf_counter()

            if "bytes" in message and message["bytes"] is not None:
                image_bytes = message["bytes"]
            else:
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
            if save_snapshots and has_violation(result):
                result["snapshot_path"] = save_violation_snapshot(image_bytes, snapshot_dir, frame_id)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return
