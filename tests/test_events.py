"""Tests for the event logging module and review API.

Run with:
    pytest tests/test_events.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _event_env(monkeypatch, tmp_path):
    """Isolate every test in its own EVENT_OUTPUT_DIR and reset the singleton."""
    out_dir = tmp_path / "events"
    monkeypatch.setenv("EVENT_LOG_ENABLED", "true")
    monkeypatch.setenv("EVENT_SNAPSHOT_ENABLED", "true")
    monkeypatch.setenv("EVENT_CONFIDENCE_THRESHOLD", "0.6")
    monkeypatch.setenv("EVENT_SNAPSHOT_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("EVENT_OUTPUT_DIR", str(out_dir))

    from src.utils import events

    events.reset_event_store_for_tests()
    yield out_dir
    events.reset_event_store_for_tests()


def _sample_result(label: str = "no_mask", confidence: float = 0.95) -> dict:
    return {
        "detections": [
            {
                "class_id": 2,
                "class_name": label,
                "confidence": confidence,
                "bbox": [10.0, 20.0, 110.0, 220.0],
            }
        ],
        "counts": {"correct_mask": 0, "incorrect_mask": 0, "no_mask": 1},
        "latency_ms": 12.5,
        "frame_width": 640,
        "frame_height": 480,
    }


def test_event_record_is_written_for_violation(_event_env):
    from src.utils.events import emit_events_from_result, get_event_store

    result = _sample_result()
    image_bytes = b"\xff\xd8\xff\xd9"  # tiny fake JPEG payload

    emitted = emit_events_from_result(
        result,
        source="image",
        image_bytes=image_bytes,
        client_id="test-client",
    )

    assert len(emitted) == 1
    event = emitted[0]
    assert event["label"] == "no_mask"
    assert event["confidence"] == 0.95
    assert event["bbox"] == [10.0, 20.0, 110.0, 220.0]
    assert event["frame_width"] == 640
    assert event["frame_height"] == 480
    assert event["client_id"] == "test-client"
    assert event["snapshot_path"] is not None
    assert Path(event["snapshot_path"]).exists()
    assert result["events"] == [event["event_id"]]

    log_path = get_event_store().log_path
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted["event_id"] == event["event_id"]


def test_low_confidence_is_skipped(_event_env):
    from src.utils.events import emit_events_from_result

    result = _sample_result(confidence=0.4)
    emitted = emit_events_from_result(result, source="image", image_bytes=b"x")
    assert emitted == []
    assert "events" not in result


def test_non_violation_label_is_skipped(_event_env):
    from src.utils.events import emit_events_from_result

    result = _sample_result(label="correct_mask", confidence=0.99)
    result["counts"] = {"correct_mask": 1, "incorrect_mask": 0, "no_mask": 0}
    emitted = emit_events_from_result(result, source="image", image_bytes=b"x")
    assert emitted == []


def test_cooldown_suppresses_repeats(monkeypatch, _event_env):
    monkeypatch.setenv("EVENT_SNAPSHOT_COOLDOWN_SECONDS", "60")
    from src.utils import events

    events.reset_event_store_for_tests()

    first = events.emit_events_from_result(
        _sample_result(), source="image", image_bytes=b"x", client_id="c1"
    )
    second = events.emit_events_from_result(
        _sample_result(), source="image", image_bytes=b"x", client_id="c1"
    )
    assert len(first) == 1
    assert second == []


def test_list_and_get_event_via_api(_event_env):
    from fastapi.testclient import TestClient

    from src.api.events_routes import router as events_router
    from src.utils.events import emit_events_from_result
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(events_router)
    client = TestClient(app)

    emitted = emit_events_from_result(
        _sample_result(),
        source="image",
        image_bytes=b"\xff\xd8\xff\xd9",
        client_id="alice",
    )
    event_id = emitted[0]["event_id"]

    listed = client.get("/api/v1/events?limit=10").json()
    assert listed["count"] == 1
    assert listed["events"][0]["event_id"] == event_id

    filtered = client.get("/api/v1/events?label=no_mask&client_id=alice").json()
    assert filtered["count"] == 1

    miss = client.get("/api/v1/events?label=correct_mask").json()
    assert miss["count"] == 0

    detail = client.get(f"/api/v1/events/{event_id}").json()
    assert detail["event_id"] == event_id

    snap = client.get(f"/api/v1/events/{event_id}/snapshot")
    assert snap.status_code == 200
    assert snap.headers["content-type"].startswith("image/")
    assert snap.content == b"\xff\xd8\xff\xd9"


def test_snapshot_path_404_when_missing_on_disk(_event_env):
    from fastapi.testclient import TestClient

    from src.api.events_routes import router as events_router
    from src.utils.events import emit_events_from_result
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(events_router)
    client = TestClient(app)

    emitted = emit_events_from_result(
        _sample_result(), source="image", image_bytes=b"x"
    )
    event_id = emitted[0]["event_id"]
    Path(emitted[0]["snapshot_path"]).unlink()

    response = client.get(f"/api/v1/events/{event_id}/snapshot")
    assert response.status_code == 404
