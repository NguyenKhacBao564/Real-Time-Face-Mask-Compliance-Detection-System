from datetime import datetime
from pathlib import Path


def has_violation(result: dict) -> bool:
    counts = result.get("counts", {})
    return bool(counts.get("incorrect_mask", 0) or counts.get("no_mask", 0))


def save_violation_snapshot(image_bytes: bytes, snapshot_dir: str, frame_id: int) -> str:
    directory = Path(snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"violation_{timestamp}_frame_{frame_id}.jpg"
    path.write_bytes(image_bytes)
    return str(path)
