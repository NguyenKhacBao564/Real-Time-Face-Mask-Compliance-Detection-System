"""StreamManager — tracks active multi-stream WebSocket connections.

Each connected client is registered as a stream with a unique ID.  The manager
is fully async-safe so it can be called from concurrent WebSocket handlers.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from src.api.metrics import active_streams


@dataclass
class StreamState:
    stream_id: str
    frame_count: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    connected_at: float = field(default_factory=time.perf_counter)
    status: str = "active"
    # Opaque reference to the WebSocket so the batch worker can send results.
    websocket: Any = field(default=None, repr=False)

    @property
    def avg_latency_ms(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return round(self.total_latency_ms / self.frame_count, 2)

    def record_latency(self, latency_ms: float) -> None:
        self.frame_count += 1
        self.last_latency_ms = latency_ms
        self.total_latency_ms += latency_ms


class StreamManager:
    """Async-safe registry of active streams."""

    def __init__(self) -> None:
        self._streams: dict[str, StreamState] = {}
        self._lock = asyncio.Lock()

    async def register_stream(self, stream_id: str, websocket: Any = None) -> StreamState:
        async with self._lock:
            state = StreamState(stream_id=stream_id, websocket=websocket)
            self._streams[stream_id] = state
            return state

    async def remove_stream(self, stream_id: str) -> StreamState | None:
        async with self._lock:
            state = self._streams.pop(stream_id, None)
        if state is not None:
            state.status = "disconnected"
            active_streams.dec()
        return state

    async def list_active(self) -> list[StreamState]:
        async with self._lock:
            return [s for s in self._streams.values() if s.status == "active"]

    async def get_stream(self, stream_id: str) -> StreamState | None:
        async with self._lock:
            return self._streams.get(stream_id)

    @property
    def active_count(self) -> int:
        # Intentionally lock-free for fast logging; occasional stale read is fine.
        return sum(1 for s in self._streams.values() if s.status == "active")
