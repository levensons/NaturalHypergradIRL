import os
import threading
import time

import psutil


class PeakRAMMonitor:
    def __init__(self, interval: float = 0.05):
        if interval <= 0:
            raise ValueError(f"`interval` must be positive, got {interval}.")

        self.interval = interval
        self.process = psutil.Process(os.getpid())

        self.start_rss = 0
        self.peak_rss = 0
        self.elapsed_seconds = 0.0

        self._started_at = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _update(self) -> None:
        rss = self.process.memory_info().rss
        now = time.monotonic()

        with self._lock:
            self.peak_rss = max(self.peak_rss, rss)
            self.elapsed_seconds = now - self._started_at

    def _current_metrics(self) -> dict[str, float]:
        with self._lock:
            start_rss = self.start_rss
            peak_rss = self.peak_rss
            elapsed_seconds = self.elapsed_seconds

        return {
            "start_rss_mb": start_rss / 1024**2,
            "peak_rss_mb": peak_rss / 1024**2,
            "peak_rss_increase_mb": (peak_rss - start_rss) / 1024**2,
            "elapsed_seconds": elapsed_seconds,
        }

    def _monitor(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._update()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("PeakRAMMonitor is already running.")

        rss = self.process.memory_info().rss
        now = time.monotonic()

        with self._lock:
            self.start_rss = rss
            self.peak_rss = rss
            self.elapsed_seconds = 0.0

        self._started_at = now
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._monitor,
            name="peak-ram-monitor",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> dict[str, float]:
        if self._thread is None:
            raise RuntimeError("PeakRAMMonitor is not running.")

        self._update()
        return self._current_metrics()

    def stop(self) -> dict[str, float]:
        if self._thread is None:
            raise RuntimeError("PeakRAMMonitor is not running.")

        self._stop_event.set()
        self._thread.join()

        self._update()
        metrics = self._current_metrics()

        self._thread = None
        return metrics
