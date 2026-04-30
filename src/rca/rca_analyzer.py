# src/rca/rca_analyzer.py

import csv
import os
import time


class RCAAnalyzer:
    def __init__(self, baseline_model, z_threshold=3.0, csv_path="rca_results.csv"):
        self.baseline_model = baseline_model
        self.z_threshold = z_threshold
        self.csv_path = csv_path

        # create file with header if not exists
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "root_cause", "confidence", "features"])

    def analyze(self, features: dict) -> dict:
        anomalies = self.baseline_model.score(features)

        root_cause = self._infer_root_cause(features, anomalies)
        confidence = self._compute_confidence(anomalies)

        result = {
            "anomalies": anomalies,
            "root_cause": root_cause,
            "confidence": confidence
        }

        # log to CSV
        self._log_to_csv(features, result)

        return result

    # --------------------------------------------------
    # Root Cause Logic (CPU + GPU)
    # --------------------------------------------------

    def _infer_root_cause(self, features, anomalies):
        if not anomalies:
            return "healthy"

        cpu_flag = any("cpu" in k for k in anomalies)
        mem_flag = any("memory" in k for k in anomalies)
        sys_flag = any("system" in k for k in anomalies)
        gpu_flag = any("gpu" in k for k in anomalies)

        # ---- CPU ----
        if cpu_flag and not gpu_flag:
            return "CPU bottleneck"

        # ---- Memory ----
        if mem_flag and not gpu_flag:
            return "Memory pressure"

        # ---- GPU ----
        gpu = features.get("gpu", {})

        if gpu:
            util = gpu.get("utilization", {}).get("avg", 0)
            mem = gpu.get("memory", {}).get("avg", 0)
            temp = gpu.get("temperature", {}).get("max", 0)
            pcie_tx = gpu.get("pcie_tx", {}).get("avg", 0)
            pcie_rx = gpu.get("pcie_rx", {}).get("avg", 0)

            if util > 90:
                return "GPU bottleneck"

            if mem > 0 and util < 20:
                return "GPU memory pressure"

            if temp > 85:
                return "GPU thermal throttling"

            if pcie_tx > 1e7 or pcie_rx > 1e7:
                return "PCIe bottleneck"

        # ---- Mixed ----
        if cpu_flag and gpu_flag:
            return "CPU-GPU imbalance"

        if sys_flag:
            return "System overload / scheduling contention"

        return "Unknown anomaly"

    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    def _compute_confidence(self, anomalies):
        if not anomalies:
            return 1.0

        z_scores = [v["z_score"] for v in anomalies.values()]
        return min(1.0, max(z_scores) / 5.0)

    # --------------------------------------------------
    # CSV Logging
    # --------------------------------------------------

    def _log_to_csv(self, features, result):
        row = [
            time.time(),
            result["root_cause"],
            result["confidence"],
            self._flatten_dict(features)
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _flatten_dict(self, d, parent=""):
        """
        Flatten nested dict into single-level dict
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent}.{k}" if parent else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))

        return dict(items)