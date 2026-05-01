# src/main.py

import threading
import time
import os
import sys

from src.monitors.collector import MetricsCollector
from src.monitors.collectors.cpu_collector import CPUCollector
from src.monitors.collectors.gpu_collector import GPUCollector
from src.monitors.aggregator import RollingBuffer, Aggregator
from src.launchers.vllm_launcher import VLLMLauncher
from src.rca.baseline_model import BaselineModel
from src.rca.rca_analyzer import RCAAnalyzer

from src.utils.fault_injection.fault_orchestrator import FaultOrchestrator

# ---- Config ----
SAMPLING_INTERVAL = 0.1   # 100 ms
RCA_INTERVAL = 10         # seconds
WINDOW_SIZE = 10          # seconds

BASELINE_DURATION = 120   # 2 minutes (reduce for testing)

MODEL_NAME = "ibm-granite/granite-4.1-3b"


def main():
    launcher = None

    try:
        # ---- Step 1: Launch vLLM ----
        print("🚀 Launching vLLM...")
        launcher = VLLMLauncher(model_name=MODEL_NAME)

        info = launcher.launch()
        launcher.wait_until_ready()

        pgid = info["pgid"]

        print(f"✅ vLLM started | PID={info['pid']} PGID={pgid}\n")

    except Exception as e:
        print(f"⚠️ Failed to launch vLLM: {e}")
        print("👉 Falling back to monitoring current process\n")
        pgid = os.getpgid(os.getpid())

    # ---- Step 2: Initialize collectors ----
    collectors = [
        CPUCollector(pgid),
        GPUCollector(device_index=1)
    ]

    metrics_collector = MetricsCollector(collectors)

    # ---- Step 3: Buffer + Aggregator ----
    buffer = RollingBuffer(WINDOW_SIZE)
    aggregator = Aggregator()

    # ---- Step 4: Baseline + RCA ----
    baseline_model = BaselineModel()
    rca_analyzer = RCAAnalyzer(baseline_model)


    baseline_phase = True
    baseline_start_time = time.time()

    last_rca_time = time.time()

    print("📊 Monitoring started...\n")

    # Fault orchestrator (runs in background)
    fault_orchestrator = FaultOrchestrator()
    fault_thread = threading.Thread(target=fault_orchestrator.run, daemon=True)
    fault_thread.start()

    try:
        while True:
            loop_start = time.time()

            # ---- Collect ----
            snapshot = metrics_collector.collect()
            buffer.add(snapshot)

            # ---- RCA Trigger ----
            if time.time() - last_rca_time >= RCA_INTERVAL:

                features = aggregator.compute(buffer)

                # -------------------------------
                # BASELINE PHASE
                # -------------------------------
                if baseline_phase:
                    baseline_model.add(features)

                    elapsed = time.time() - baseline_start_time
                    print(f"📘 Baseline phase: {int(elapsed)} / {BASELINE_DURATION} sec")

                    if elapsed >= BASELINE_DURATION:
                        baseline_model.build()
                        baseline_phase = False
                        print("\n✅ Baseline established\n")

                # -------------------------------
                # RCA PHASE
                # -------------------------------
                else:
                    result = rca_analyzer.analyze(features)

                    print("\n🔍 RCA Result:")
                    print(result)

                last_rca_time = time.time()

            # ---- Maintain 100 ms loop ----
            elapsed = time.time() - loop_start
            time.sleep(max(0, SAMPLING_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")

    finally:
        if launcher and launcher.is_running():
            print("🔻 Stopping vLLM...")
            launcher.stop()
            launcher.wait()

        print("✅ Clean exit")


if __name__ == "__main__":
    main()