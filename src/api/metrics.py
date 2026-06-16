from prometheus_client import Counter, Gauge, Histogram

# COUNTERS
frames_processed_total = Counter(
    "frames_processed_total",
    "Total number of frames processed",
    ["stream_id"]
)
inference_batches_total = Counter(
    "inference_batches_total",
    "Total number of inference batches processed"
)
frames_dropped_total = Counter(
    "frames_dropped_total",
    "Total number of frames dropped"
)

# GAUGES
active_streams = Gauge(
    "active_streams",
    "Number of currently active video streams"
)
queue_depth = Gauge(
    "queue_depth",
    "Current number of frames waiting in the processing queue"
)
current_batch_size = Gauge(
    "current_batch_size",
    "Size of the current or last processed batch"
)

# GPU GAUGES
gpu_utilization_percent = Gauge(
    "gpu_utilization_percent",
    "GPU Utilization in percent"
)
gpu_memory_used_mb = Gauge(
    "gpu_memory_used_mb",
    "GPU Memory Used in MB"
)

# HISTOGRAMS
buckets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Latency of inference processing per frame",
    buckets=buckets
)
batch_latency_seconds = Histogram(
    "batch_latency_seconds",
    "Latency of inference processing per batch",
    buckets=buckets
)
end_to_end_latency_seconds = Histogram(
    "end_to_end_latency_seconds",
    "End-to-end latency from frame ingestion to result delivery",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
)
