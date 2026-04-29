import argparse
import math
import statistics
import time
from pathlib import Path

from src.inference.predictor import MaskDetector


def benchmark_images(image_dir: Path, predictor: MaskDetector) -> dict:
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    predictor.predict_image(str(image_paths[0]))

    latencies: list[float] = []
    for image_path in image_paths:
        started = time.perf_counter()
        predictor.predict_image(str(image_path))
        latencies.append((time.perf_counter() - started) * 1000)

    avg_latency = statistics.mean(latencies)
    sorted_latencies = sorted(latencies)
    p95_index = min(len(sorted_latencies) - 1, math.ceil(len(sorted_latencies) * 0.95) - 1)
    return {
        "images": len(image_paths),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(sorted_latencies[p95_index], 2),
        "fps": round(1000 / avg_latency, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory of benchmark images")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    predictor = MaskDetector(model_path=args.model, image_size=args.imgsz)
    print(benchmark_images(Path(args.images), predictor))


if __name__ == "__main__":
    main()
