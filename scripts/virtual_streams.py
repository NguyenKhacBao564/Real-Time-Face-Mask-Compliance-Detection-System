#!/usr/bin/env python3
"""Virtual camera simulator for multi-stream testing.

Spawns N concurrent WebSocket clients that each read from the same input video
but start at different frame offsets so the streams are not identical.  Each
client sends frames to /api/v1/ws/multi-detect at a target FPS and prints the
detections it receives back.

Usage:
    python scripts/virtual_streams.py --video path/to/video.mp4 --streams 8 --fps 15
    python scripts/virtual_streams.py --video path/to/video.mp4 --streams 8 --fps 15 --server ws://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import cv2
import websockets


def _read_video_frames(video_path: str) -> list[bytes]:
    """Read all frames from a video file as JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[bytes] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode(".jpg", frame)
        frames.append(jpeg.tobytes())
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    return frames


async def run_client(
    client_id: int,
    all_frames: list[bytes],
    frame_offset: int,
    server_url: str,
    target_fps: float,
    total_cycles: int,
) -> dict:
    """Run a single virtual camera client.

    Connects to the multi-stream WebSocket, sends frames in a loop, and collects
    detection results.  Returns a summary dict.
    """
    frame_interval = 1.0 / target_fps
    total_frames_sent = 0
    total_frames_received = 0
    latencies_ms: list[float] = []
    detections_summary: dict[str, int] = {}
    errors = 0

    ws_url = f"{server_url}/api/v1/ws/multi-detect?client_id=virtual_{client_id}"
    stream_label = f"[stream-{client_id}]"

    try:
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
            print(f"{stream_label} connected  (offset={frame_offset}, "
                  f"frames_in_video={len(all_frames)}, cycles={total_cycles})")

            receive_task = asyncio.create_task(
                _receive_loop(ws, stream_label, latencies_ms, detections_summary)
            )

            for _cycle in range(total_cycles):
                for i in range(len(all_frames)):
                    frame_idx = (i + frame_offset) % len(all_frames)
                    jpeg = all_frames[frame_idx]

                    t0 = time.perf_counter()
                    await ws.send(jpeg)
                    total_frames_sent += 1

                    elapsed = time.perf_counter() - t0
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

            # Give a moment for remaining responses to arrive.
            await asyncio.sleep(1.0)
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

    except Exception as exc:
        errors += 1
        print(f"{stream_label} ERROR: {exc}")

    total_frames_received = len(latencies_ms)
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    summary = {
        "client_id": client_id,
        "frames_sent": total_frames_sent,
        "frames_received": total_frames_received,
        "avg_latency_ms": round(avg_latency, 2),
        "detections": detections_summary,
        "errors": errors,
    }
    print(
        f"{stream_label} done — sent={total_frames_sent}  received={total_frames_received}  "
        f"avg_latency={avg_latency:.1f}ms  detections={detections_summary}"
    )
    return summary


async def _receive_loop(
    ws,
    label: str,
    latencies_ms: list[float],
    detections_summary: dict[str, int],
) -> None:
    """Background task that reads detection results from the WebSocket."""
    while True:
        try:
            msg = await ws.recv()
            data = json.loads(msg)
            if "error" in data:
                continue
            latency = data.get("latency_ms", 0)
            latencies_ms.append(latency)
            for det in data.get("detections", []):
                cls = det.get("class_name", "unknown")
                detections_summary[cls] = detections_summary.get(cls, 0) + 1
        except asyncio.CancelledError:
            break
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual multi-stream test client")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--streams", type=int, default=8, help="Number of virtual streams (default: 8)")
    parser.add_argument("--fps", type=float, default=15, help="Target FPS per stream (default: 15)")
    parser.add_argument("--server", default="ws://localhost:8000", help="WebSocket server URL")
    parser.add_argument("--cycles", type=int, default=3, help="How many times each client loops through the video")
    args = parser.parse_args()

    print(f"Loading video frames from {args.video} ...")
    frames = _read_video_frames(args.video)
    print(f"Loaded {len(frames)} frames")

    offset_step = max(1, len(frames) // args.streams)

    print(f"\nLaunching {args.streams} virtual streams at {args.fps} FPS, {args.cycles} cycles each\n")

    tasks = [
        run_client(
            client_id=i,
            all_frames=frames,
            frame_offset=(i * offset_step) % len(frames),
            server_url=args.server,
            target_fps=args.fps,
            total_cycles=args.cycles,
        )
        for i in range(args.streams)
    ]

    async def run_all():
        return await asyncio.gather(*tasks, return_exceptions=True)

    summaries = asyncio.run(run_all())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_sent = 0
    total_received = 0
    for s in summaries:
        if isinstance(s, Exception):
            print(f"  Client failed: {s}")
            continue
        total_sent += s["frames_sent"]
        total_received += s["frames_received"]

    print(f"  Total frames sent:     {total_sent}")
    print(f"  Total results recv:    {total_received}")
    print(f"  Active streams:        {args.streams}")
    print("=" * 60)


if __name__ == "__main__":
    main()
