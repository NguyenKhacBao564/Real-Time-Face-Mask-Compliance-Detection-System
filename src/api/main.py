import asyncio
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.events_routes import router as events_router
from src.api.predictor_provider import get_predictor, preload_predictor
from src.api.routes import router as api_router
from src.api.websocket import router as websocket_router
from src.multistream.batch_worker import _WorkItem, batch_worker_loop
from src.multistream.routes import router as multistream_router, configure as configure_multistream
from src.multistream.stream_manager import StreamManager
from src.api.gpu_monitor import monitor_gpu_loop
from src.utils.config import load_app_config
from src.utils.logger import get_logger

logger = get_logger("main")

app = FastAPI(
    title="Real-Time Face Mask Compliance Detection System",
    version="0.1.0",
)

app.include_router(api_router)
app.include_router(events_router)
app.include_router(websocket_router)
app.include_router(multistream_router)

frontend_dir = Path("frontend/simple_webcam_client")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Multi-stream shared state.
_stream_manager = StreamManager()
_frame_queue: asyncio.Queue[_WorkItem | None] = asyncio.Queue()


@app.on_event("startup")
def startup() -> None:
    if os.getenv("PRELOAD_MODEL", "0").lower() in {"1", "true", "yes"}:
        preload_predictor()

    # Start the batched inference worker.
    config = load_app_config()
    max_batch = int(os.getenv("MULTI_MAX_BATCH", "8"))
    collect_timeout_ms = int(os.getenv("MULTI_COLLECT_TIMEOUT_MS", "20"))

    configure_multistream(_stream_manager, _frame_queue)

    model = get_predictor()._load_model()

    asyncio.get_event_loop().create_task(
        batch_worker_loop(
            queue=_frame_queue,
            stream_manager=_stream_manager,
            model=model,
            image_size=int(config["model"]["image_size"]),
            conf=float(config["model"]["confidence_threshold"]),
            iou=float(config["model"]["iou_threshold"]),
            max_batch=max_batch,
            collect_timeout_s=collect_timeout_ms / 1000.0,
        )
    )
    asyncio.get_event_loop().create_task(monitor_gpu_loop())
    logger.info(
        "multi-stream worker started max_batch=%d collect_timeout_ms=%d",
        max_batch,
        collect_timeout_ms,
    )


@app.get("/", response_model=None)
def index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Face mask detection API is running"}
