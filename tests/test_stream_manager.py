"""Tests for StreamManager — register, remove, list_active."""

import asyncio
import pytest

from src.multistream.stream_manager import StreamManager


@pytest.fixture
def manager():
    return StreamManager()


@pytest.mark.asyncio
async def test_register_and_list(manager):
    await manager.register_stream("s1")
    await manager.register_stream("s2")
    active = await manager.list_active()
    ids = {s.stream_id for s in active}
    assert ids == {"s1", "s2"}


@pytest.mark.asyncio
async def test_remove_stream(manager):
    await manager.register_stream("s1")
    removed = await manager.remove_stream("s1")
    assert removed is not None
    assert removed.stream_id == "s1"
    assert removed.status == "disconnected"
    active = await manager.list_active()
    assert len(active) == 0


@pytest.mark.asyncio
async def test_remove_nonexistent_returns_none(manager):
    result = await manager.remove_stream("nope")
    assert result is None


@pytest.mark.asyncio
async def test_list_active_excludes_disconnected(manager):
    await manager.register_stream("s1")
    await manager.register_stream("s2")
    await manager.remove_stream("s1")
    active = await manager.list_active()
    assert len(active) == 1
    assert active[0].stream_id == "s2"


@pytest.mark.asyncio
async def test_get_stream(manager):
    await manager.register_stream("s1")
    state = await manager.get_stream("s1")
    assert state is not None
    assert state.stream_id == "s1"
    assert state.status == "active"


@pytest.mark.asyncio
async def test_get_stream_not_found(manager):
    state = await manager.get_stream("nope")
    assert state is None


@pytest.mark.asyncio
async def test_record_latency(manager):
    await manager.register_stream("s1")
    state = await manager.get_stream("s1")
    state.record_latency(10.0)
    state.record_latency(20.0)
    assert state.frame_count == 2
    assert state.last_latency_ms == 20.0
    assert state.avg_latency_ms == 15.0


@pytest.mark.asyncio
async def test_active_count(manager):
    assert manager.active_count == 0
    await manager.register_stream("s1")
    await manager.register_stream("s2")
    assert manager.active_count == 2
    await manager.remove_stream("s1")
    assert manager.active_count == 1


@pytest.mark.asyncio
async def test_register_replaces_existing(manager):
    await manager.register_stream("s1")
    await manager.register_stream("s1")  # re-register same id
    active = await manager.list_active()
    # Should have exactly one entry (the latest registration).
    assert len(active) == 1
    assert active[0].stream_id == "s1"
