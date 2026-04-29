from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router as api_router
from src.api.websocket import router as websocket_router


app = FastAPI(
    title="Real-Time Face Mask Compliance Detection System",
    version="0.1.0",
)

app.include_router(api_router)
app.include_router(websocket_router)

frontend_dir = Path("frontend/simple_webcam_client")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", response_model=None)
def index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Face mask detection API is running"}
