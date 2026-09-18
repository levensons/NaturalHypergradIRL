import os
import threading
import time
from dataclasses import dataclass

import psutil


_MB = 1024 ** 2


@dataclass
class PeakRSSResult:
    rss_before_mb: float
    rss_after_mb: float
    rss_peak_mb: float
    rss_peak_delta_mb: float
    elapsed_sec: float


class PeakRSSProfiler:
    def __init__(self, interval_sec: float = 0.005):
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())

        self._stop_event = threading.Event()
        self._thread = None

        self._rss_before = 0
        self._rss_after = 0
        self._rss_peak = 0
        self._start_time = 0.0
        self._elapsed = 0.0

    def _sample(self):
        while not self._stop_event.is_set():
            rss = self.process.memory_info().rss
            self._rss_peak = max(self._rss_peak, rss)
            self._stop_event.wait(self.interval_sec)

    def __enter__(self):
        self._stop_event.clear()

        self._rss_before = self.process.memory_info().rss
        self._rss_peak = self._rss_before
        self._start_time = time.perf_counter()

        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

        self._elapsed = time.perf_counter() - self._start_time
        self._rss_after = self.process.memory_info().rss
        self._rss_peak = max(self._rss_peak, self._rss_after)

    def result(self) -> PeakRSSResult:
        return PeakRSSResult(
            rss_before_mb=self._rss_before / _MB,
            rss_after_mb=self._rss_after / _MB,
            rss_peak_mb=self._rss_peak / _MB,
            rss_peak_delta_mb=(self._rss_peak - self._rss_before) / _MB,
            elapsed_sec=self._elapsed,
        )
