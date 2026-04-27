from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict, Optional

try:
    import pynvml
except ImportError as exc:
    pynvml = None  # type: ignore[assignment]
    _pynvml_import_error = exc
else:
    _pynvml_import_error = None


from src.metadata import Metadata

def _safe(fn, default=None):
    if pynvml is None:
        return default
    try:
        return fn()
    except Exception:
        return default


def _to_mb(value: Optional[int]) -> Optional[float]:
    return None if value is None else round(value / (1024 * 1024), 2)


def _to_watts(value: Optional[int]) -> Optional[float]:
    return None if value is None else round(value / 1000.0, 3)


def _require_pynvml() -> None:
    if pynvml is None:
        raise RuntimeError(
            "pynvml is required. Install with: pip install nvidia-ml-py3"
        ) from _pynvml_import_error

class GpuMetricsCollector:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata
        self.gpu_index = getattr(metadata, "gpu_index", None)
        self.gpu_handle = None
        self.running = False
        self.lock = threading.Lock()
        self.metrics: Dict[str, list] = {
            "gpu_utils": [],
            "gpu_mems": [],
            "powers": [],
            "temps": [],
            "sm_clocks": [],
            "pcie_txs": [],
            "pcie_rxs": [],
        }

    async def collect_metrics(self, duration_s: float = 1.0, interval_s: float = 0.01):
        _require_pynvml()
        pynvml.nvmlInit()
        try:
            index = 0 if self.gpu_index is None else int(self.gpu_index)
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            _safe(lambda: pynvml.nvmlShutdown())
            raise

    def _shutdown(self) -> None:
        if pynvml is None:
            return
        _safe(lambda: pynvml.nvmlShutdown())

    def _poll_loop(self, interval_s: float) -> None:
        if pynvml is None or self.gpu_handle is None:
            return

        while self.running:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000
                temp = pynvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                sm_clock = pynvml.nvmlDeviceGetClockInfo(
                    self.gpu_handle, pynvml.NVML_CLOCK_SM
                )

                try:
                    pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                        self.gpu_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                        self.gpu_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                except Exception:
                    pcie_tx, pcie_rx = 0, 0

                with self.lock:
                    self.metrics["gpu_utils"].append(getattr(util, "gpu", None))
                    self.metrics["gpu_mems"].append(getattr(mem, "used", None))
                    self.metrics["powers"].append(power)
                    self.metrics["temps"].append(temp)
                    self.metrics["sm_clocks"].append(sm_clock)
                    self.metrics["pcie_txs"].append(pcie_tx)
                    self.metrics["pcie_rxs"].append(pcie_rx)
            except Exception:
                pass

            time.sleep(interval_s)

    async def collect_metrics(self, duration_s: float = 1.0, interval_s: float = 0.01):
        collected_at = time.time()

        try:
            self._init_handle()
        except Exception as exc:
            return {
                "gpu_index": 0 if self.gpu_index is None else self.gpu_index,
                "error": str(exc),
                "collected_at": collected_at,
            }

        self.running = True
        t = threading.Thread(target=self._poll_loop, args=(interval_s,), daemon=True)
        t.start()

        try:
            await asyncio.sleep(duration_s)
        finally:
            self.running = False
            t.join(timeout=1.0)
            self._shutdown()

        with self.lock:
            metrics_snapshot = {k: list(v) for k, v in self.metrics.items()}

        return {
            "gpu_index": 0 if self.gpu_index is None else self.gpu_index,
            "collected_at": collected_at,
            "duration_s": duration_s,
            "interval_s": interval_s,
            **metrics_snapshot,
        }


def collect_gpu_metrics() -> Dict[str, Any]:
    _require_pynvml()
    pynvml.nvmlInit()
    try:
        timestamp = time.time()
        count = _safe(lambda: pynvml.nvmlDeviceGetCount(), 0) or 0
        gpus = []

        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = _safe(lambda: pynvml.nvmlDeviceGetName(h), b"")
            uuid = _safe(lambda: pynvml.nvmlDeviceGetUUID(h), b"")
            mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
            util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))

            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="ignore")

            sm_clock = _safe(
                lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            )

            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                    h, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    h, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
            except pynvml.NVMLError:
                pcie_tx, pcie_rx = 0, 0

            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "uuid": uuid,
                    "timestamp": timestamp,
                    "temperature_c": _safe(
                        lambda: pynvml.nvmlDeviceGetTemperature(
                            h, pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    "utilization_gpu_pct": getattr(util, "gpu", None),
                    "utilization_mem_pct": getattr(util, "memory", None),
                    "memory_total_mb": _to_mb(getattr(mem, "total", None)),
                    "memory_used_mb": _to_mb(getattr(mem, "used", None)),
                    "memory_free_mb": _to_mb(getattr(mem, "free", None)),
                    "power_w": _to_watts(_safe(lambda: pynvml.nvmlDeviceGetPowerUsage(h))),
                    "fan_speed_pct": _safe(lambda: pynvml.nvmlDeviceGetFanSpeed(h)),
                    "pstate": _safe(lambda: pynvml.nvmlDeviceGetPerformanceState(h)),
                    "sm_clock_mhz": sm_clock,
                    "pcie_tx_bytes_per_sec": pcie_tx,
                    "pcie_rx_bytes_per_sec": pcie_rx,
                }
            )

        return {"timestamp": timestamp, "gpu_count": count, "gpus": gpus}
    finally:
        _safe(lambda: pynvml.nvmlShutdown())


if __name__ == "__main__":
    print(json.dumps(collect_gpu_metrics(), indent=2))
