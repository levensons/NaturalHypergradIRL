import os
import threading
import time

import psutil
from mlflow.tracking import MlflowClient


class PeakRAMMonitor:
    def __init__(
        self,
        interval: float = 0.05,
        log_every: float = 30.0,
        run_id: str | None = None,
    ):
        if interval <= 0:
            raise ValueError(f"`interval` must be positive, got {interval}.")

        if log_every <= 0:
            raise ValueError(f"`log_every` must be positive, got {log_every}.")

        self.interval = interval
        self.log_every = log_every
        self.run_id = run_id

        self.process = psutil.Process(os.getpid())
        self.mlflow_client = MlflowClient() if run_id is not None else None

        self.start_rss = 0
        self.peak_rss = 0
        self.elapsed_seconds = 0.0

        self._started_at = 0.0
        self._last_logged_at = 0.0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

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

    def _log_to_mlflow(self) -> None:
        if self.mlflow_client is None or self.run_id is None:
            return

        metrics = self._current_metrics()
        timestamp = int(time.time() * 1000)
        step = int(metrics["elapsed_seconds"])

        try:
            self.mlflow_client.log_metric(
                run_id=self.run_id,
                key="resources/training_peak_rss_mb",
                value=metrics["peak_rss_mb"],
                timestamp=timestamp,
                step=step,
            )
            self.mlflow_client.log_metric(
                run_id=self.run_id,
                key="resources/training_peak_rss_increase_mb",
                value=metrics["peak_rss_increase_mb"],
                timestamp=timestamp,
                step=step,
            )
            self.mlflow_client.log_metric(
                run_id=self.run_id,
                key="resources/training_elapsed_seconds",
                value=metrics["elapsed_seconds"],
                timestamp=timestamp,
                step=step,
            )
        except Exception:
            # Resource monitoring must never kill the training process.
            pass

    def _monitor(self) -> None:
        while not self._stop_event.wait(self.interval):
            rss = self.process.memory_info().rss
            now = time.monotonic()

            with self._lock:
                self.peak_rss = max(self.peak_rss, rss)
                self.elapsed_seconds = now - self._started_at

            if now - self._last_logged_at >= self.log_every:
                self._log_to_mlflow()
                self._last_logged_at = now

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
        self._last_logged_at = now

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._monitor,
            name="peak-ram-monitor",
            daemon=True,
        )
        self._thread.start()

        self._log_to_mlflow()

    def snapshot(self) -> dict[str, float]:
        rss = self.process.memory_info().rss
        now = time.monotonic()

        with self._lock:
            self.peak_rss = max(self.peak_rss, rss)
            self.elapsed_seconds = now - self._started_at

        self._log_to_mlflow()

        return self._current_metrics()

    def stop(self) -> dict[str, float]:
        if self._thread is None:
            raise RuntimeError("PeakRAMMonitor is not running.")

        self._stop_event.set()
        self._thread.join()

        metrics = self.snapshot()
        self._thread = None

        return metrics
