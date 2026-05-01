# src/rca/rca_analyzer.py

import csv
import os
import time


class RCAAnalyzer:
    """
    Performs:
    - Z-score anomaly detection
    - Root cause inference
    - CSV logging
    """

    def __init__(self, baseline_model, z_threshold=5.0, csv_path="rca_results.csv"):
        self.baseline_model = baseline_model
        self.z_threshold = z_threshold
        self.csv_path = csv_path

        # Initialize CSV
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "status",
                    "root_cause",
                    "confidence",
                    "anomalies",
                    "features"
                ])

    # --------------------------------------------------
    # MAIN ANALYSIS
    # --------------------------------------------------

    def analyze(self, features: dict) -> dict:
        anomalies = self._compute_z_scores(features)

        root_cause = self._infer_root_cause(anomalies)
        confidence = self._compute_confidence(anomalies)

        status = "healthy" if not anomalies else "unhealthy"

        result = {
            "status": status,
            "root_cause": root_cause,
            "confidence": confidence,
            "anomalies": anomalies
        }

        self._log_to_csv(features, result)

        return result

    # --------------------------------------------------
    # Z-SCORE ANOMALY DETECTION
    # --------------------------------------------------

    def _compute_z_scores(self, features):
        anomalies = {}

        for key, stats in self.baseline_model.baseline.items():
            try:
                current = self._get_value(features, key)
            except Exception:
                continue

            mean = stats["mean"]
            std = stats["std"]

            if std == 0:
                continue

            z = (current - mean) / std

            if abs(z) > self.z_threshold:
                anomalies[key] = {
                    "value": current,
                    "z_score": z
                }

        return anomalies

    # --------------------------------------------------
    # ROOT CAUSE INFERENCE
    # --------------------------------------------------

    def _infer_root_cause(self, anomalies):
        if not anomalies:
            return "healthy"

        cpu_flag = any("cpu" in k for k in anomalies)
        gpu_flag = any("gpu" in k for k in anomalies)
        mem_flag = any("memory" in k for k in anomalies)
        sys_flag = any("system" in k for k in anomalies)

        # CPU issues
        if cpu_flag and not gpu_flag:
            return "CPU bottleneck"

        # GPU issues
        if gpu_flag and not cpu_flag:
            return "GPU bottleneck"

        # Memory issues
        if mem_flag:
            return "Memory pressure"

        # System contention
        if sys_flag:
            return "System contention"

        # Mixed issue
        if cpu_flag and gpu_flag:
            return "CPU-GPU imbalance"

        return "Unknown anomaly"

    # --------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------

    def _compute_confidence(self, anomalies):
        if not anomalies:
            return 1.0

        z_scores = [abs(v["z_score"]) for v in anomalies.values()]
        return min(1.0, max(z_scores) / 5.0)

    # --------------------------------------------------
    # CSV LOGGING
    # --------------------------------------------------

    def _log_to_csv(self, features, result):
        flattened_features = self._flatten_dict(features)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                time.time(),
                result["status"],
                result["root_cause"],
                result["confidence"],
                str(result["anomalies"]),
                str(flattened_features)
            ])

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def _flatten_dict(self, d, parent=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent}.{k}" if parent else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def _get_value(self, d, key):
        parts = key.split(".")
        val = d
        for p in parts:
            val = val[p]
        return val