#!/usr/bin/env python3
"""Export YOLOv8 model to ONNX formats (FP32 and FP16)."""
import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    # Determine model path
    model_path = os.environ.get("MODEL_PATH", "mask_yolov8n_pwmfd_baseline/weights/best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please set MODEL_PATH environment variable or ensure the default path exists.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    # 1. Export FP32 (Dynamic shapes)
    print("\n>>> Exporting to ONNX FP32 (dynamic)...")
    path_fp32 = model.export(format="onnx", dynamic=True, half=False, simplify=True)
    dest_fp32 = output_dir / "best.onnx"
    shutil.move(path_fp32, dest_fp32)
    print(f"Saved to {dest_fp32}")

    # 2. Export FP16 (Dynamic shapes)
    print("\n>>> Exporting to ONNX FP16 (dynamic)...")
    try:
        # Note: half=True for ONNX usually requires GPU
        device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") or shutil.which("nvidia-smi") else "cpu"
        path_fp16 = model.export(format="onnx", dynamic=True, half=True, simplify=True, device=device)
        dest_fp16 = output_dir / "best_fp16.onnx"
        shutil.move(path_fp16, dest_fp16)
        print(f"Saved to {dest_fp16}")
    except Exception as e:
        print(f"Warning: FP16 export failed: {e}")
        print("Note: FP16 export typically requires an NVIDIA GPU.")
        dest_fp16 = None

    # Compare file sizes
    pt_size = Path(model_path).stat().st_size / (1024 * 1024)
    onnx_fp32_size = dest_fp32.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*30)
    print("EXPORT SUMMARY")
    print("="*30)
    print(f"PyTorch (.pt):  {pt_size:6.2f} MB")
    print(f"ONNX (FP32):    {onnx_fp32_size:6.2f} MB")
    if dest_fp16 and dest_fp16.exists():
        onnx_fp16_size = dest_fp16.stat().st_size / (1024 * 1024)
        print(f"ONNX (FP16):    {onnx_fp16_size:6.2f} MB")
    print("="*30)

if __name__ == "__main__":
    main()
