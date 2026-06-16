"""Batched inference worker — producer/consumer pattern for multi-stream detection.

Architecture:
  ALL WebSocket streams push (stream_id, jpeg_bytes) into a single asyncio.Queue.
  ONE background worker loop drains the queue, collects up to `max_batch` frames,
  runs YOLO inference on the ENTIRE batch in a single GPU call, then routes each
  result back to the correct stream via WebSocket.

Why batching raises GPU utilization:
  GPUs are massively parallel — a single frame (640×640×3) only saturates a
  fraction of CUDA cores.  By stacking N frames into one tensor (N×3×640×640),
  the same CUDA kernels process N images with nearly the same wall-clock time
  as one image (plus a small linear overhead for data transfer).  This means
  throughput scales roughly N× while per-frame latency only increases ~20-40%
  compared to single-image inference, giving much higher total GPU utilization
  across multiple camera streams.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.multistream.stream_manager import StreamManager
from src.utils.logger import get_logger

logger = get_logger("batch_worker")


@dataclass
class _WorkItem:
    stream_id: str
    jpeg_bytes: bytes
    enqueue_time: float


async def collect_batch(
    queue: asyncio.Queue[_WorkItem | None],
    max_batch: int,
    timeout_s: float,
) -> list[_WorkItem]:
    """Drain the queue into a batch, waiting up to *timeout_s* for the first item.

    Returns as soon as the batch is full OR the timeout expires (whichever comes
    first).  A ``None`` sentinel signals shutdown.
    """
    batch: list[_WorkItem] = []

    # Block until at least one item arrives (or sentinel).
    try:
        item = await asyncio.wait_for(queue.get(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return batch

    if item is None:
        return batch  # sentinel — worker should stop
    batch.append(item)

    # Greedily drain whatever else is ready, up to max_batch.
    while len(batch) < max_batch:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if item is None:
            break
        batch.append(item)

    return batch


def _run_batch_inference(
    jpeg_batch: list[bytes],
    model: Any,
    image_size: int,
    conf: float,
    iou: float,
) -> list[dict]:
    """Run YOLO on a batch of JPEG byte arrays.

    Ultralytics ``predict()`` accepts a list of numpy arrays directly, which
    avoids the temp-file-per-frame disk I/O of the single-stream path.
    """
    import cv2

    # Decode JPEG bytes → numpy arrays (BGR, as expected by OpenCV / Ultralytics).
    frames: list[np.ndarray] = []
    for jpeg_bytes in jpeg_batch:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            # Corrupt frame — substitute a blank 640×640 image so batch size
            # is preserved (YOLO requires consistent input shapes).
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        frames.append(img)

    # model.predict() with a list of arrays runs a single batched GPU call.
    results = model.predict(
        source=frames,
        imgsz=image_size,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    from src.inference.postprocess import count_detections, parse_ultralytics_result

    parsed: list[dict] = []
    for res in results:
        detections = parse_ultralytics_result(res)
        orig_shape = getattr(res, "orig_shape", None)
        fh = int(orig_shape[0]) if orig_shape is not None and len(orig_shape) >= 2 else None
        fw = int(orig_shape[1]) if orig_shape is not None and len(orig_shape) >= 2 else None
        parsed.append({
            "detections": detections,
            "counts": count_detections(detections),
            "frame_width": fw,
            "frame_height": fh,
        })
    return parsed


async def batch_worker_loop(
    queue: asyncio.Queue[_WorkItem | None],
    stream_manager: StreamManager,
    model: Any,
    *,
    image_size: int = 640,
    conf: float = 0.35,
    iou: float = 0.45,
    max_batch: int = 8,
    collect_timeout_s: float = 0.020,
) -> None:
    """Background coroutine: collects batches, runs GPU inference, sends results.

    Runs forever until a ``None`` sentinel is placed in *queue*.
    """
    logger.info(
        "batch_worker started max_batch=%d collect_timeout_ms=%d",
        max_batch,
        int(collect_timeout_s * 1000),
    )

    while True:
        batch = await collect_batch(queue, max_batch, collect_timeout_s)
        if not batch:
            continue
        if batch[0] is None:
            break

        # Check for sentinel inside the batch (from partial drain).
        real_items = [b for b in batch if b is not None]
        if not real_items:
            break

        batch_size = len(real_items)
        jpeg_batch = [item.jpeg_bytes for item in real_items]

        t0 = time.perf_counter()
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,  # default ThreadPoolExecutor
                _run_batch_inference,
                jpeg_batch,
                model,
                image_size,
                conf,
                iou,
            )
        except Exception as exc:
            logger.error("batch inference failed batch_size=%d error=%s", batch_size, exc)
            # Send error to all clients in this batch.
            for item in real_items:
                stream = await stream_manager.get_stream(item.stream_id)
                if stream and stream.websocket:
                    try:
                        await stream.websocket.send_json({"error": str(exc)})
                    except Exception:
                        pass
            continue

        batch_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        q_depth = queue.qsize()

        inference_batches_total.inc()
        batch_latency_seconds.observe(batch_latency_ms / 1000.0)
        current_batch_size.set(batch_size)
        metrics_queue_depth.set(q_depth)

        logger.info(
            "batch_inferred batch_size=%d batch_latency_ms=%.2f "
            "queue_depth=%d avg_per_frame_ms=%.2f",
            batch_size,
            batch_latency_ms,
            q_depth,
            round(batch_latency_ms / batch_size, 2),
        )

        # Route each result back to the originating stream.
        for item, result in zip(real_items, results):
            latency_sec = time.perf_counter() - item.enqueue_time
            latency_ms = round(latency_sec * 1000, 2)
            
            frames_processed_total.labels(stream_id=item.stream_id).inc()
            inference_latency_seconds.observe((batch_latency_ms / batch_size) / 1000.0)
            end_to_end_latency_seconds.observe(latency_sec)
            
            result["stream_id"] = item.stream_id
            result["latency_ms"] = latency_ms
            result["batch_latency_ms"] = batch_latency_ms

            stream = await stream_manager.get_stream(item.stream_id)
            if stream:
                stream.record_latency(latency_ms)
                if stream.websocket:
                    try:
                        await stream.websocket.send_json(result)
                    except Exception:
                        pass

    logger.info("batch_worker stopped")
