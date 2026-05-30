import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.events_routes import router as events_router
from src.api.predictor_provider import preload_predictor
from src.api.routes import router as api_router
from src.api.websocket import router as websocket_router


app = FastAPI(
    title="Real-Time Face Mask Compliance Detection System",
    version="0.1.0",
)

app.include_router(api_router)
app.include_router(events_router)
app.include_router(websocket_router)

frontend_dir = Path("frontend/simple_webcam_client")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.on_event("startup")
def startup() -> None:
    if os.getenv("PRELOAD_MODEL", "0").lower() in {"1", "true", "yes"}:
        preload_predictor()


@app.get("/", response_model=None)
def index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Face mask detection API is running"}
