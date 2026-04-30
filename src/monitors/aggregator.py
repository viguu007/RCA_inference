# src/monitors/aggregator.py

from collections import deque
import time
import statistics


class RollingBuffer:
    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.buffer = deque()

    def add(self, snapshot: dict):
        self.buffer.append(snapshot)
        self._evict_old()

    def _evict_old(self):
        current_time = time.time()
        while self.buffer and (current_time - self.buffer[0]["timestamp"] > self.window_seconds):
            self.buffer.popleft()

    def get_all(self):
        return list(self.buffer)


class Aggregator:
    def __init__(self, cpu_spike_threshold=80, gpu_spike_threshold=90):
        self.cpu_spike_threshold = cpu_spike_threshold
        self.gpu_spike_threshold = gpu_spike_threshold

    def compute(self, buffer: RollingBuffer) -> dict:
        data = buffer.get_all()

        if not data:
            return {}

        # ---- CPU ----
        cpu_values = []
        memory_values = []
        running_procs = []
        ctx_switches = []

        # ---- GPU ----
        gpu_utils = []
        gpu_memory = []
        gpu_power = []
        gpu_temp = []
        gpu_sm_clock = []
        pcie_tx = []
        pcie_rx = []

        for snap in data:
            metrics = snap.get("metrics", {})

            # ---- CPU ----
            cpu = metrics.get("cpu", {})
            pg = cpu.get("process_group", {})
            sys = cpu.get("system", {})

            cpu_values.append(pg.get("cpu_percent", 0))
            memory_values.append(pg.get("memory_rss_bytes", 0))
            running_procs.append(sys.get("num_running_processes", 0))
            ctx_switches.append(sys.get("context_switches", 0))

            # ---- GPU ----
            gpu = metrics.get("gpu", {})
            if "error" not in gpu:
                gpu_utils.append(gpu.get("utilization_percent", 0))
                gpu_memory.append(gpu.get("memory_used_bytes", 0))
                gpu_power.append(gpu.get("power_watts", 0))
                gpu_temp.append(gpu.get("temperature_c", 0))
                gpu_sm_clock.append(gpu.get("sm_clock_mhz", 0))
                pcie_tx.append(gpu.get("pcie_tx_bytes", 0))
                pcie_rx.append(gpu.get("pcie_rx_bytes", 0))

        # ---- CPU Aggregation ----
        cpu_features = self._aggregate_series(cpu_values, self.cpu_spike_threshold)
        mem_features = self._aggregate_series(memory_values)
        proc_avg = self._safe_mean(running_procs)

        ctx_delta = 0
        if len(ctx_switches) > 1:
            ctx_delta = ctx_switches[-1] - ctx_switches[0]

        # ---- GPU Aggregation ----
        gpu_features = {}
        if gpu_utils:
            gpu_features = {
                "utilization": self._aggregate_series(gpu_utils, self.gpu_spike_threshold),
                "memory": self._aggregate_series(gpu_memory),
                "power": self._aggregate_series(gpu_power),
                "temperature": self._aggregate_series(gpu_temp),
                "sm_clock": self._aggregate_series(gpu_sm_clock),
                "pcie_tx": self._aggregate_series(pcie_tx),
                "pcie_rx": self._aggregate_series(pcie_rx),
            }

        return {
            "cpu": {
                "usage": cpu_features,
                "memory": mem_features,
            },
            "system": {
                "avg_running_processes": proc_avg,
                "context_switch_delta": ctx_delta,
            },
            "gpu": gpu_features
        }

    # ---- Helpers ----

    def _aggregate_series(self, values, spike_threshold=None):
        if not values:
            return {}

        avg = self._safe_mean(values)
        mx = max(values)
        std = self._safe_std(values)

        spikes = 0
        if spike_threshold is not None:
            spikes = sum(1 for v in values if v > spike_threshold)

        return {
            "avg": avg,
            "max": mx,
            "std": std,
            "spike_count": spikes
        }

    def _safe_mean(self, values):
        return statistics.mean(values) if values else 0

    def _safe_std(self, values):
        return statistics.stdev(values) if len(values) > 1 else 0