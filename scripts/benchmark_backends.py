#!/usr/bin/env python3
"""Benchmark PyTorch vs ONNX Runtime backends on GPU."""
import os
import sys
import time
import json
import subprocess
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    ort = None

def get_gpu_info():
    """Get GPU details via nvidia-smi."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name,driver_version,memory.total", "--format=csv,noheader"],
            encoding="utf-8"
        )
        return res.strip()
    except Exception:
        return "NVIDIA GPU (details unavailable)"

def benchmark_pytorch(model_path, batch_sizes, img_size=640, iterations=200, warmup=20):
    print(f"\n>>> Benchmarking PyTorch FP32 (CUDA)...")
    if not torch.cuda.is_available():
        print("Error: CUDA not available for PyTorch.")
        return []
    
    model = YOLO(model_path).to("cuda")
    results = []

    for bs in batch_sizes:
        print(f"  Batch size {bs}...", end=" ", flush=True)
        # Random input batch
        dummy_input = torch.randn(bs, 3, img_size, img_size).to("cuda")

        # Warmup
        for _ in range(warmup):
            model(dummy_input, verbose=False)
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(dummy_input, verbose=False)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000) # ms
        
        mean_lat = np.mean(latencies)
        fps = (bs * 1000) / mean_lat
        mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        results.append({
            "backend": "PyTorch FP32",
            "batch_size": bs,
            "fps": fps,
            "lat_mean": mean_lat,
            "lat_p50": np.percentile(latencies, 50),
            "lat_p95": np.percentile(latencies, 95),
            "lat_p99": np.percentile(latencies, 99),
            "mem_mb": mem
        })
        print(f"Done. FPS: {fps:.1f}")
    
    return results

def benchmark_ort(model_path, batch_sizes, img_size=640, iterations=200, warmup=20):
    print(f"\n>>> Benchmarking ONNX Runtime GPU FP16...")
    if ort is None:
        print("  Error: onnxruntime-gpu not installed. Please run: pip install onnxruntime-gpu")
        return []
    
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        print("  Error: CUDAExecutionProvider not available in ONNX Runtime.")
        return []

    # Load session
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    results = []

    for bs in batch_sizes:
        print(f"  Batch size {bs}...", end=" ", flush=True)
        # Random input batch (FP16)
        dummy_input = np.random.randn(bs, 3, img_size, img_size).astype(np.float16)

        # Warmup
        for _ in range(warmup):
            session.run(None, {input_name: dummy_input})
        
        torch.cuda.synchronize() # Still use torch for GPU sync
        
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000) # ms
        
        mean_lat = np.mean(latencies)
        fps = (bs * 1000) / mean_lat
        
        # Memory tracking for ORT via torch is limited but gives a baseline
        mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        results.append({
            "backend": "ORT GPU FP16",
            "batch_size": bs,
            "fps": fps,
            "lat_mean": mean_lat,
            "lat_p50": np.percentile(latencies, 50),
            "lat_p95": np.percentile(latencies, 95),
            "lat_p99": np.percentile(latencies, 99),
            "mem_mb": mem
        })
        print(f"Done. FPS: {fps:.1f}")
        
    return results

def format_markdown(gpu_info, results):
    lines = [
        "# Inference Benchmark Results",
        "",
        f"- **GPU:** {gpu_info}",
        f"- **CUDA available:** {torch.cuda.is_available()}",
        f"- **PyTorch version:** {torch.__version__}",
        f"- **ONNX Runtime version:** {ort.__version__ if ort else 'N/A'}",
        "",
        "| Backend | Batch | FPS | Mean Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) | GPU Mem (MB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in results:
        lines.append(
            f"| {r['backend']} | {r['batch_size']} | {r['fps']:>8.1f} | {r['lat_mean']:>8.2f} | {r['lat_p50']:>8.2f} | {r['lat_p95']:>8.2f} | {r['lat_p99']:>8.2f} | {r['mem_mb']:>8.1f} |"
        )
    return "\n".join(lines)

def main():
    pt_path = os.environ.get("MODEL_PATH", "mask_yolov8n_pwmfd_baseline/weights/best.pt")
    onnx_path = "models/best_fp16.onnx"
    
    if not os.path.exists(pt_path):
        print(f"Error: model {pt_path} not found.")
        sys.exit(1)
        
    batch_sizes = [1, 4, 8, 16]
    all_results = []
    
    # 1. PyTorch
    all_results.extend(benchmark_pytorch(pt_path, batch_sizes))
    
    # 2. ONNX Runtime
    if os.path.exists(onnx_path):
        all_results.extend(benchmark_ort(onnx_path, batch_sizes))
    else:
        print(f"\nWarning: {onnx_path} not found. Run scripts/export_onnx.py first.")

    # Generate outputs
    gpu_info = get_gpu_info()
    md_table = format_markdown(gpu_info, all_results)
    
    # Print to stdout
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(md_table)
    
    # Save to files
    with open("benchmark_results.json", "w") as f:
        json.dump({"gpu": gpu_info, "results": all_results}, f, indent=2)
    
    with open("BENCHMARK.md", "w") as f:
        f.write(md_table)
    
    print(f"\nResults saved to benchmark_results.json and BENCHMARK.md")

if __name__ == "__main__":
    main()
