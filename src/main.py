# src/main.py

import time
import os
import signal
import sys

from src.monitors.collector import MetricsCollector
from src.monitors.collectors.cpu_collector import CPUCollector
from src.monitors.collectors.gpu_collector import GPUCollector
from src.monitors.aggregator import RollingBuffer, Aggregator
from src.launchers.vllm_launcher import VLLMLauncher
from src.rca.baseline_model import BaselineModel
from src.rca.rca_analyzer import RCAAnalyzer
from src.rca import baseline_model


# ---- Config ----
SAMPLING_INTERVAL = 0.1   # 100 ms
RCA_INTERVAL = 20         # seconds
WINDOW_SIZE = 20          # seconds

MODEL_NAME = "facebook/opt-125m"  # change if needed


def dummy_provenance(features):
    return features


def dummy_rca(features):
    cpu = features.get("cpu", {})
    if cpu.get("avg_percent", 0) > 80:
        print("⚠️ RCA: CPU bottleneck detected")
    else:
        print("✅ RCA: System healthy")


def main():
    launcher = None

    try:
        # ---- Step 1: Launch vLLM ----
        print("🚀 Launching vLLM...")
        launcher = VLLMLauncher(model_name=MODEL_NAME)

        info = launcher.launch()
        pgid = info["pgid"]

        print(f"✅ vLLM started | PID={info['pid']} PGID={pgid}\n")

    except Exception as e:
        print(f"⚠️ Failed to launch vLLM: {e}")
        print("👉 Falling back to monitoring current process\n")
        pgid = os.getpgid(os.getpid())

    # ---- Step 2: Initialize collectors ----
    collectors = [
        CPUCollector(pgid),
        GPUCollector(device_index=0)
    ]

    metrics_collector = MetricsCollector(collectors)

    # ---- Step 3: Buffer + Aggregator ----
    buffer = RollingBuffer(WINDOW_SIZE)
    aggregator = Aggregator()

    baseline_model = BaselineModel()
    rca_analyzer = RCAAnalyzer(baseline_model)

    last_rca_time = time.time()

    print("📊 Monitoring started...\n")

    try:
        while True:
            loop_start = time.time()

            # 1. Collect snapshot
            snapshot = metrics_collector.collect()

            # 2. Add to buffer
            buffer.add(snapshot)

            # 3. RCA trigger
            if time.time() - last_rca_time >= RCA_INTERVAL:

                features = aggregator.compute(buffer)

                result = rca_analyzer.analyze(features)

                last_rca_time = time.time()

            # 4. Maintain sampling interval
            elapsed = time.time() - loop_start
            time.sleep(max(0, SAMPLING_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        print("🔻 Stopping vLLM...")
        launcher.stop()
        launcher.wait()
        print("✅ Clean exit")

    finally:
        if launcher and launcher.is_running():
            print("🔻 Stopping vLLM...")
            launcher.stop()
            launcher.wait()

        print("✅ Clean exit")


if __name__ == "__main__":
    main()