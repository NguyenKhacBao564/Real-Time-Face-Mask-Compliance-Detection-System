# Inference Benchmark Results

- **GPU:** NVIDIA GeForce RTX 3090 Ti, 550.163.01, 24564 MiB
- **CUDA available:** True
- **PyTorch version:** 2.12.0+cu126
- **ONNX Runtime version:** 1.26.0

| Backend | Batch | FPS | Mean Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) | GPU Mem (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PyTorch FP32 | 1 |    255.0 |     3.92 |     3.91 |     3.98 |     4.13 |     69.1 |
| PyTorch FP32 | 4 |    807.7 |     4.95 |     4.95 |     4.97 |     5.03 |    144.1 |
| PyTorch FP32 | 8 |    891.9 |     8.97 |     8.96 |     9.03 |     9.19 |    247.1 |
| PyTorch FP32 | 16 |    727.8 |    21.98 |    23.63 |    23.84 |    23.93 |    447.6 |
| ORT GPU FP16 | 1 |    542.8 |     1.84 |     1.84 |     1.87 |     1.88 |    447.6 |
| ORT GPU FP16 | 4 |   1020.1 |     3.92 |     3.86 |     4.17 |     4.29 |    447.6 |
| ORT GPU FP16 | 8 |   1240.8 |     6.45 |     6.42 |     6.68 |     6.84 |    447.6 |
| ORT GPU FP16 | 16 |   1347.2 |    11.88 |    11.86 |    12.00 |    12.18 |    447.6 |