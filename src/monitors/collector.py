# src/monitors/collector.py

from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class MetricsCollector:
    """
    Orchestrates all collectors.
    - Runs them in parallel per sampling tick
    - Produces a unified snapshot
    """

    def __init__(self, collectors):
        """
        collectors: list of BaseCollector instances
        """
        self.collectors = collectors

    def collect(self) -> dict:
        """
        Collect metrics from all collectors in parallel.

        Returns:
        {
            "timestamp": float,
            "metrics": {
                "cpu": {...},
                "gpu": {...}
            }
        }
        """

        snapshot = {
            "timestamp": time.time(),
            "metrics": {}
        }

        # Run all collectors in parallel
        with ThreadPoolExecutor(max_workers=len(self.collectors)) as executor:
            future_to_name = {
                executor.submit(collector.collect): collector.name()
                for collector in self.collectors
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]

                try:
                    result = future.result()
                except Exception as e:
                    # Fail-safe: don’t crash the pipeline
                    result = {
                        "error": str(e)
                    }

                snapshot["metrics"][name] = result

        return snapshot