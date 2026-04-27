
class metricsCollected:
    def __init__(self):
        self.cpu_usage_percent = list()
        self.memory_usage_percent = list()
        self.gpu_utils = list()

    def clear(self):
        self.cpu_usage_percent.clear()
        self.memory_usage_percent.clear()
        self.gpu_utils.clear()
