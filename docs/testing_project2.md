# Manual Test Guide — System & App Improvements

Quick `curl` recipes for the new AI-camera-style event/review APIs. Assumes the
server is running on `http://localhost:8000`. Adjust the port if you use a
different one (Cloud Run / Docker uses `8080` by default).

```bash
export BASE=http://localhost:8000
```

## 1. Run the server locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

Optional environment overrides (defaults shown):

```bash
EVENT_LOG_ENABLED=true \
EVENT_CONFIDENCE_THRESHOLD=0.6 \
EVENT_SNAPSHOT_ENABLED=true \
EVENT_SNAPSHOT_COOLDOWN_SECONDS=3 \
EVENT_OUTPUT_DIR=outputs/events \
uvicorn src.api.main:app --reload
```

## 2. Health check

```bash
curl -s $BASE/health
# {"status":"ok"}
```

## 3. Image inference (will emit events on violation)

```bash
curl -s -X POST "$BASE/api/v1/predict/image?client_id=demo" \
  -F "file=@C.jpeg" | jq
```

Expected (abridged):

```json
{
  "detections": [
    {"class_name": "no_mask", "confidence": 0.91, "bbox": [...]}
  ],
  "counts": {"correct_mask": 0, "incorrect_mask": 0, "no_mask": 1},
  "latency_ms": 42.1,
  "frame_width": 1280,
  "frame_height": 720,
  "events": ["b1f3...e9c"]
}
```

If `events` is present, an event row was appended to
`outputs/events/events.jsonl` and (if snapshots enabled) a JPEG was written to
`outputs/events/snapshots/`.

## 4. List events

```bash
curl -s "$BASE/api/v1/events?limit=10" | jq
curl -s "$BASE/api/v1/events?label=no_mask&limit=5" | jq
curl -s "$BASE/api/v1/events?client_id=demo&limit=5" | jq
curl -s "$BASE/api/v1/events?start_time=2024-01-01T00:00:00Z" | jq
```

## 5. Inspect a single event

```bash
EVENT_ID=$(curl -s "$BASE/api/v1/events?limit=1" | jq -r '.events[0].event_id')
curl -s "$BASE/api/v1/events/$EVENT_ID" | jq
```

## 6. Download the snapshot

```bash
curl -s "$BASE/api/v1/events/$EVENT_ID/snapshot" -o /tmp/violation.jpg
file /tmp/violation.jpg
```

## 7. Drive the WebSocket from the browser

Open `http://localhost:8000/` in a browser. The Violations panel now polls the
event API every 3 seconds and shows real persisted events with a clickable
`snapshot` link.

## 8. Run the automated tests

```bash
pytest tests/test_events.py -v
```

Covers: event record creation, confidence threshold, cooldown, list/detail/
snapshot endpoints, and the missing-snapshot 404 path.
