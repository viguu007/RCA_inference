import statistics


class BaselineModel:
    def __init__(self):
        self.data = []
        self.baseline = {}

    def add(self, features: dict):
        self.data.append(features)

    def build(self):
        """
        Build baseline statistics from collected feature windows
        """
        if not self.data:
            return

        keys = self._flatten_keys(self.data[0])

        for key in keys:
            values = [self._get_value(d, key) for d in self.data]

            if len(values) > 1:
                self.baseline[key] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values)
                }
            else:
                self.baseline[key] = {
                    "mean": values[0],
                    "std": 0
                }

    def score(self, features: dict):
        """
        Compare new features against baseline
        """
        anomalies = {}

        for key, stats in self.baseline.items():
            current = self._get_value(features, key)
            mean = stats["mean"]
            std = stats["std"]

            if std == 0:
                continue

            z = abs((current - mean) / std)

            if z > 3:  # threshold
                anomalies[key] = {
                    "value": current,
                    "z_score": z
                }

        return anomalies

    def _flatten_keys(self, d, parent=""):
        keys = []
        for k, v in d.items():
            new_key = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                keys.extend(self._flatten_keys(v, new_key))
            else:
                keys.append(new_key)
        return keys

    def _get_value(self, d, key):
        parts = key.split(".")
        val = d
        for p in parts:
            val = val[p]
        return val