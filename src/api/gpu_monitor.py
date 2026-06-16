import asyncio
import logging
from src.api.metrics import gpu_utilization_percent, gpu_memory_used_mb

logger = logging.getLogger("gpu_monitor")

async def monitor_gpu_loop(interval_s: float = 2.0):
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            logger.warning("No NVIDIA GPUs found via pynvml.")
            return
            
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        logger.info("Started GPU monitoring for device 0")
        
        while True:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_utilization_percent.set(utilization.gpu)
                gpu_memory_used_mb.set(memory_info.used / (1024 * 1024))
            except Exception as e:
                logger.error(f"Error reading GPU metrics: {e}")
                
            await asyncio.sleep(interval_s)
    except ImportError:
        logger.warning("pynvml not installed, skipping GPU metrics monitoring.")
    except Exception as e:
        logger.warning(f"Failed to initialize pynvml: {e}. Skipping GPU metrics monitoring.")
