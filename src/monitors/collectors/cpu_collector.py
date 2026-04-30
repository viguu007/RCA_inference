# src/monitors/collectors/cpu_collector.py

from src.monitors.collectors.base_collector import BaseCollector
from src.monitors.process_tracker import get_processes_in_group
import psutil


class CPUCollector(BaseCollector):
    def __init__(self, pgid: int):
        self.pgid = pgid
        self._warmup_done = False

    def name(self) -> str:
        return "cpu"

    def _get_cpu_percent_safe(self, proc: psutil.Process) -> float:
        try:
            return proc.cpu_percent(interval=0.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    def _get_memory_safe(self, proc: psutil.Process) -> int:
        """
        Returns RSS memory in bytes
        """
        try:
            return proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _warmup(self, processes):
        for proc in processes:
            try:
                proc.cpu_percent(interval=0.0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def collect(self) -> dict:
        processes = get_processes_in_group(self.pgid)

        # Warm-up phase
        if not self._warmup_done:
            self._warmup(processes)
            self._warmup_done = True

            return {
                "process_group": {
                    "cpu_percent": 0.0,
                    "memory_rss_bytes": 0,
                    "num_processes": len(processes),
                },
                "system": {
                    "num_running_processes": len(psutil.pids()),
                    "context_switches": psutil.cpu_stats().ctx_switches,
                },
            }

        total_cpu = 0.0
        total_memory = 0
        count = 0

        for proc in processes:
            total_cpu += self._get_cpu_percent_safe(proc)
            total_memory += self._get_memory_safe(proc)
            count += 1

        # System-wide stats
        cpu_stats = psutil.cpu_stats()

        return {
            "process_group": {
                "cpu_percent": total_cpu,
                "memory_rss_bytes": total_memory,
                "num_processes": count,
            },
            "system": {
                "num_running_processes": len(psutil.pids()),
                "context_switches": cpu_stats.ctx_switches,
            },
        }