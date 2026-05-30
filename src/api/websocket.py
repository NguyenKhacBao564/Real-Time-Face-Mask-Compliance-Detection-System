import tempfile
import time
import uuid

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.api.predictor_provider import get_predictor
from src.utils.events import emit_events_from_result
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger("ws")


@router.websocket("/api/v1/ws/detect")
async def detect_websocket(
    websocket: WebSocket,
    client_id: str | None = Query(None),
) -> None:
    await websocket.accept()
    session_id = client_id or uuid.uuid4().hex[:12]
    session_started = time.perf_counter()
    frame_id = 0
    total_violations = 0
    logger.info("ws_connect session_id=%s", session_id)

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
                    await websocket.send_json(
                        {"error": str(exc), "frame_id": frame_id}
                    )
                    continue

            request_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            result["frame_id"] = frame_id
            result["latency_ms"] = round(request_latency_ms, 2)

            emitted = emit_events_from_result(
                result,
                source="websocket",
                image_bytes=image_bytes,
                client_id=session_id,
                session_id=session_id,
            )
            total_violations += len(emitted)

            # Backward-compatible: keep snapshot_path on the response when a
            # violation snapshot was just written. Older clients still see it.
            if emitted and emitted[0].get("snapshot_path"):
                result["snapshot_path"] = emitted[0]["snapshot_path"]

            await websocket.send_json(result)

            if frame_id % 50 == 0:
                elapsed = max(1e-6, time.perf_counter() - session_started)
                fps = frame_id / elapsed
                logger.info(
                    "ws_progress session_id=%s frames=%d fps=%.2f "
                    "last_inference_ms=%s last_request_ms=%s violations_emitted=%d",
                    session_id,
                    frame_id,
                    fps,
                    result.get("latency_ms"),
                    request_latency_ms,
                    total_violations,
                )
    except WebSocketDisconnect:
        return
    finally:
        elapsed = max(1e-6, time.perf_counter() - session_started)
        fps = frame_id / elapsed if frame_id else 0.0
        logger.info(
            "ws_disconnect session_id=%s frames=%d elapsed_s=%.2f fps=%.2f "
            "violations_emitted=%d",
            session_id,
            frame_id,
            elapsed,
            fps,
            total_violations,
        )
