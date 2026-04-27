
from src.metadata import Metadata
import asyncio
import time
import psutil

class CpuMetricsCollector:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata

    async def collect_metrics(self):
        process_id = getattr(self.metadata, "process_id", None)
        if process_id is None:
            raise ValueError("metadata.process_id is required")

        try:
            process = psutil.Process(process_id)
            cpu_usage_percent = await asyncio.to_thread(process.cpu_percent, interval=None)
            process_mem_usage_percent = await asyncio.to_thread(process.memory_percent)

            return {
                "process_id": process_id,
                "cpu_usage_percent": cpu_usage_percent,
                "memory_usage_percent": process_mem_usage_percent,
                "collected_at": time.time(),
            }
        except psutil.AccessDenied:
            print(f"Access denied for process {process_id}. EXITING from program.")
            exit(1)

        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            return {
                "process_id": process_id,
                "error": str(exc),
                "collected_at": time.time(),
            }