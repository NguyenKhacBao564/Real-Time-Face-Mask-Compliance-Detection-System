import argparse
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

    latencies: list[float] = []
    for image_path in image_paths:
        started = time.perf_counter()
        predictor.predict_image(str(image_path))
        latencies.append((time.perf_counter() - started) * 1000)

    avg_latency = statistics.mean(latencies)
    return {
        "images": len(image_paths),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2)
        if len(latencies) >= 20
        else round(max(latencies), 2),
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

