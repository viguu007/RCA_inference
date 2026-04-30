# src/monitors/collectors/base_collector.py

from abc import ABC, abstractmethod


class BaseCollector(ABC):
    """
    Abstract base class for all metric collectors.
    Each collector must implement:
      - name(): unique identifier for the metric
      - collect(): returns collected metric data
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the collector (e.g., 'cpu', 'gpu').
        """
        pass

    @abstractmethod
    def collect(self):
        """
        Collects metrics and returns data (preferably a dict).
        """
        pass