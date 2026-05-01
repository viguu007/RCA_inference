# src/utils/fault_orchestrator.py

import time
import random
import csv
import os

from src.utils.fault_injection.fault_injector import FaultInjector


class FaultOrchestrator:
    def __init__(
        self,
        csv_path="fault_log.csv",
        min_duration=30,
        max_duration=100,
        sleep_between_faults=(120.0, 600.0),
    ):
        """
        min_duration / max_duration → how long each fault runs
        sleep_between_faults → idle time between faults
        """

        self.injector = FaultInjector()
        self.csv_path = csv_path
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sleep_between_faults = sleep_between_faults

        self._init_csv()

    # ---------------------------------------
    # CSV INIT
    # ---------------------------------------
    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "fault_type", "duration_sec"])

    # ---------------------------------------
    # RANDOM FAULT SELECTION
    # ---------------------------------------
    def _choose_fault(self):
        return random.choice(["cpu", "gpu"])

    # ---------------------------------------
    # APPLY FAULT
    # ---------------------------------------
    def _apply_fault(self, fault_type, duration):
        print(f"\n⚡ Injecting {fault_type.upper()} fault for {duration}s")

        start_time = time.time()

        if fault_type == "cpu":
            self.injector.start_cpu_stress(num_workers=4)

        elif fault_type == "gpu":
            self.injector.start_gpu_stress(duration=duration)

        # Run for duration
        time.sleep(duration)

        # Stop everything
        self.injector.stop_all()

        # Log
        self._log_fault(start_time, fault_type, duration)

    # ---------------------------------------
    # LOGGING
    # ---------------------------------------
    def _log_fault(self, timestamp, fault_type, duration):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, fault_type, duration])

    # ---------------------------------------
    # MAIN LOOP
    # ---------------------------------------
    def run(self):
        
        time.sleep(200)  # Initial delay before first fault
        print("🎯 Fault orchestrator started...\n")


        while True:
            # Random idle time before next fault
            idle_time = random.randint(*self.sleep_between_faults)
            print(f"💤 Sleeping for {idle_time}s before next fault")
            time.sleep(idle_time)

            # Choose fault
            fault_type = self._choose_fault()

            # Random duration
            duration = random.randint(self.min_duration, self.max_duration)

            # Apply fault
            self._apply_fault(fault_type, duration)