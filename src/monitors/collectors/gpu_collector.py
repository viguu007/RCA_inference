# src/monitors/collectors/gpu_collector.py

from src.monitors.collectors.base_collector import BaseCollector

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPUCollector(BaseCollector):
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.initialized = False
        self.gpu_handle = None

        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.initialized = True
            except Exception:
                self.initialized = False

    def name(self) -> str:
        return "gpu"

    def collect(self) -> dict:
        """
        Collect GPU metrics as a snapshot.
        """

        if not self.initialized:
            return {
                "error": "NVML not available"
            }

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            temp = pynvml.nvmlDeviceGetTemperature(
                self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            sm_clock = pynvml.nvmlDeviceGetClockInfo(
                self.gpu_handle, pynvml.NVML_CLOCK_SM
            )

            # PCIe throughput (may fail on some systems)
            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                    self.gpu_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    self.gpu_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
            except pynvml.NVMLError:
                pcie_tx, pcie_rx = 0, 0

            return {
                "utilization_percent": util.gpu,
                "memory_used_bytes": mem.used,
                "memory_total_bytes": mem.total,
                "power_watts": power,
                "temperature_c": temp,
                "sm_clock_mhz": sm_clock,
                "pcie_tx_bytes": pcie_tx,
                "pcie_rx_bytes": pcie_rx,
            }

        except pynvml.NVMLError as e:
            return {
                "error": str(e)
            }