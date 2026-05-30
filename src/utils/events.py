"""Event logging for face mask compliance violations.

Records every qualifying violation detection (label in {no_mask, incorrect_mask}
and confidence >= threshold) into a JSONL append-only log and optionally writes
a snapshot of the offending frame. A per-(source, label) cooldown prevents log
and snapshot flooding during a continuous violation.

The store is intentionally simple (JSONL on local disk) so it is easy to
explain in an interview and easy to ship inside a small Docker container.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger("events")

VIOLATION_LABELS = frozenset({"incorrect_mask", "no_mask"})


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with microseconds, e.g. 2024-05-30T12:34:56.789012Z."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_event_config() -> dict[str, Any]:
    """Read event-logging config from environment variables with sensible defaults."""
    return {
        "enabled": _bool_env("EVENT_LOG_ENABLED", True),
        "confidence_threshold": float(os.getenv("EVENT_CONFIDENCE_THRESHOLD", "0.6")),
        "snapshot_enabled": _bool_env("EVENT_SNAPSHOT_ENABLED", True),
        "cooldown_seconds": float(os.getenv("EVENT_SNAPSHOT_COOLDOWN_SECONDS", "3")),
        "output_dir": os.getenv("EVENT_OUTPUT_DIR", "outputs/events"),
    }


class EventStore:
    """Append-only JSONL store for violation events plus snapshot helpers.

    The cooldown map lives in-process. That is fine for a single-worker FastAPI
    deployment (the realistic case for this project). If we ever scale to many
    workers we would move cooldown + storage into Redis / SQLite.
    """

    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.output_dir = Path(output_dir)
        self.snapshot_dir = self.output_dir / "snapshots"
        self.log_path = self.output_dir / "events.jsonl"
        self._write_lock = threading.Lock()
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._cooldown_lock = threading.Lock()

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def cooldown_ok(self, source_key: str, label: str, cooldown_s: float) -> bool:
        """Return True and arm the timer if the cooldown window has elapsed."""
        if cooldown_s <= 0:
            return True
        key = (source_key, label)
        now = time.monotonic()
        with self._cooldown_lock:
            last = self._cooldowns.get(key, 0.0)
            if (now - last) < cooldown_s:
                return False
            self._cooldowns[key] = now
        return True

    def write(self, event: dict[str, Any]) -> None:
        self.ensure_dirs()
        line = json.dumps(event, ensure_ascii=False)
        with self._write_lock, self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save_snapshot(self, image_bytes: bytes, label: str, event_id: str) -> str:
        self.ensure_dirs()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.snapshot_dir / f"{ts}_{label}_{event_id[:8]}.jpg"
        path.write_bytes(image_bytes)
        return str(path)

    def list_events(
        self,
        label: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        client_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if label and row.get("label") != label:
                    continue
                if client_id and row.get("client_id") != client_id:
                    continue
                ts = row.get("timestamp", "")
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                rows.append(row)
        rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return rows[:limit]

    def get_event(self, event_id: str) -> Optional[dict[str, Any]]:
        if not self.log_path.exists():
            return None
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event_id") == event_id:
                    return row
        return None


_store_singleton: EventStore | None = None
_store_singleton_lock = threading.Lock()


def get_event_store() -> EventStore:
    """Return a process-wide EventStore, rebuilt if EVENT_OUTPUT_DIR changes."""
    global _store_singleton
    target_dir = get_event_config()["output_dir"]
    with _store_singleton_lock:
        if _store_singleton is None or str(_store_singleton.output_dir) != target_dir:
            _store_singleton = EventStore(target_dir)
    return _store_singleton


def reset_event_store_for_tests() -> None:
    """Test helper: drop the singleton so a new EVENT_OUTPUT_DIR takes effect."""
    global _store_singleton
    with _store_singleton_lock:
        _store_singleton = None


def emit_events_from_result(
    result: dict[str, Any],
    *,
    source: str,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    image_bytes: Optional[bytes] = None,
    client_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Emit a violation event per qualifying detection.

    A detection qualifies if its label is in VIOLATION_LABELS and its confidence
    is >= EVENT_CONFIDENCE_THRESHOLD. A per-(source_key, label) cooldown then
    suppresses repeat events during a continuous violation so the JSONL log and
    snapshot folder do not explode.

    Side effects:
      * appends one JSON line per emitted event
      * optionally writes a snapshot JPEG when image_bytes is provided
      * mutates ``result`` to add ``events`` (list[event_id]) when non-empty

    Returns the list of emitted event records (possibly empty).
    """
    cfg = get_event_config()
    if not cfg["enabled"]:
        return []

    detections = result.get("detections") or []
    if not detections:
        return []

    store = get_event_store()
    threshold = cfg["confidence_threshold"]
    cooldown = cfg["cooldown_seconds"]
    snapshot_enabled = cfg["snapshot_enabled"]
    cooldown_key = client_id or session_id or source

    fw = frame_width if frame_width is not None else result.get("frame_width")
    fh = frame_height if frame_height is not None else result.get("frame_height")
    latency_ms = result.get("latency_ms")

    emitted: list[dict[str, Any]] = []
    for det in detections:
        label = det.get("class_name")
        if label not in VIOLATION_LABELS:
            continue
        confidence = float(det.get("confidence", 0.0))
        if confidence < threshold:
            continue
        if not store.cooldown_ok(cooldown_key, label, cooldown):
            continue

        event_id = uuid.uuid4().hex
        snapshot_path: Optional[str] = None
        if snapshot_enabled and image_bytes is not None:
            try:
                snapshot_path = store.save_snapshot(image_bytes, label, event_id)
            except OSError as exc:
                logger.warning("snapshot write failed: %s", exc)

        event = {
            "event_id": event_id,
            "timestamp": _now_iso(),
            "source": source,
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": det.get("bbox"),
            "frame_width": fw,
            "frame_height": fh,
            "latency_ms": latency_ms,
            "snapshot_path": snapshot_path,
            "client_id": client_id,
            "session_id": session_id,
        }
        store.write(event)
        emitted.append(event)

    if emitted:
        result["events"] = [e["event_id"] for e in emitted]
        logger.info(
            "events_emitted source=%s client_id=%s count=%d labels=%s",
            source,
            client_id or session_id or "-",
            len(emitted),
            ",".join(sorted({e["label"] for e in emitted})),
        )

    return emitted
