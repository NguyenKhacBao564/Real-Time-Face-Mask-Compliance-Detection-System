"""Lightweight review API for violation events.

Endpoints (mounted under /api/v1):
  GET /events                        - list recent events with simple filters
  GET /events/{event_id}             - fetch a single event record
  GET /events/{event_id}/snapshot    - stream the snapshot JPEG, if any
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from src.utils.events import get_event_store

router = APIRouter(prefix="/api/v1")


@router.get("/events")
def list_events(
    label: Optional[str] = Query(None, description="Filter by class label"),
    start_time: Optional[str] = Query(
        None, description="ISO timestamp lower bound, e.g. 2024-05-30T00:00:00Z"
    ),
    end_time: Optional[str] = Query(None, description="ISO timestamp upper bound"),
    client_id: Optional[str] = Query(None, description="Filter by client/session id"),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    rows = get_event_store().list_events(
        label=label,
        start_time=start_time,
        end_time=end_time,
        client_id=client_id,
        limit=limit,
    )
    return {"count": len(rows), "events": rows}


@router.get("/events/{event_id}")
def get_event(event_id: str) -> dict:
    event = get_event_store().get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


@router.get("/events/{event_id}/snapshot")
def get_event_snapshot(event_id: str):
    event = get_event_store().get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    snapshot_path = event.get("snapshot_path")
    if not snapshot_path:
        raise HTTPException(status_code=404, detail="no snapshot for event")
    path = Path(snapshot_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="snapshot missing on disk")
    return FileResponse(path, media_type="image/jpeg")
