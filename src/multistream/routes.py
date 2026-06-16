"""Multi-stream WebSocket endpoint — one shared GPU batch for N camera streams.

Route: /api/v1/ws/multi-detect

Each connecting client is registered with a unique stream_id.  Incoming JPEG
frames are pushed into a shared asyncio.Queue where the batch worker collects
them, runs one batched GPU call, and routes results back to the correct client.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.multistream.batch_worker import _WorkItem
from src.multistream.stream_manager import StreamManager
from src.utils.events import emit_events_from_result
from src.utils.logger import get_logger

logger = get_logger("multi_ws")

router = APIRouter()

# Shared state — initialised by start_batch_worker() at app startup.
# These module-level references are set once and then read-only.
stream_manager: StreamManager | None = None
frame_queue: asyncio.Queue[_WorkItem | None] | None = None


def configure(
    manager: StreamManager,
    queue: asyncio.Queue[_WorkItem | None],
) -> None:
    """Called once at startup so the WebSocket handler can access shared state."""
    global stream_manager, frame_queue
    stream_manager = manager
    frame_queue = queue


@router.websocket("/api/v1/ws/multi-detect")
async def multi_detect_websocket(
    websocket: WebSocket,
    client_id: str | None = Query(None),
) -> None:
    await websocket.accept()

    if stream_manager is None or frame_queue is None:
        await websocket.send_json({"error": "multi-stream not initialised"})
        await websocket.close()
        return

    stream_id = client_id or uuid.uuid4().hex[:12]
    await stream_manager.register_stream(stream_id, websocket=websocket)
    logger.info("multi_connect stream_id=%s", stream_id)

    session_started = time.perf_counter()
    frames_received = 0

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

            if "bytes" not in message or message["bytes"] is None:
                continue

            frames_received += 1
            item = _WorkItem(
                stream_id=stream_id,
                jpeg_bytes=message["bytes"],
                enqueue_time=time.perf_counter(),
            )
            await frame_queue.put(item)

    except WebSocketDisconnect:
        pass
    finally:
        await stream_manager.remove_stream(stream_id)
        elapsed = max(1e-6, time.perf_counter() - session_started)
        logger.info(
            "multi_disconnect stream_id=%s frames=%d elapsed_s=%.2f fps=%.2f",
            stream_id,
            frames_received,
            elapsed,
            frames_received / elapsed,
        )
