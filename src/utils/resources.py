import json
import os
import threading
import time
from pathlib import Path
import mlflow
import psutil


class PeakRAMMonitor:
    def __init__(
        self,
        output_path: str | Path,
        interval: float = 0.05,
        persist_every: float = 30.0,
        log_to_mlflow: bool = True,
    ):
        if interval <= 0:
            raise ValueError(f"`interval` must be positive, got {interval}.")

        if persist_every <= 0:
            raise ValueError(f"`persist_every` must be positive, got {persist_every}.")

        self.output_path = Path(output_path)
        self.interval = interval
        self.persist_every = persist_every
        self.log_to_mlflow = log_to_mlflow

        self.process = psutil.Process(os.getpid())

        self.start_rss = 0
        self.peak_rss = 0
        self.elapsed_seconds = 0.0

        self._started_at = 0.0
        self._last_persisted_at = 0.0
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

    def _persist(self) -> None:
        metrics = self._current_metrics()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = self.output_path.with_suffix(
            self.output_path.suffix + ".tmp"
        )

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        # Atomic replacement: the output file is never partially written.
        os.replace(temporary_path, self.output_path)

        if self.log_to_mlflow and mlflow.active_run() is not None:
            try:
                mlflow.log_metrics(
                    {
                        "resources/training_peak_rss_mb":
                            metrics["peak_rss_mb"],
                        "resources/training_peak_rss_increase_mb":
                            metrics["peak_rss_increase_mb"],
                        "resources/training_elapsed_seconds":
                            metrics["elapsed_seconds"],
                    },
                    step=int(metrics["elapsed_seconds"]),
                )
            except Exception:
                # The local JSON remains the reliable fallback.
                pass

    def _monitor(self) -> None:
        while not self._stop_event.wait(self.interval):
            rss = self.process.memory_info().rss
            now = time.monotonic()

            with self._lock:
                self.peak_rss = max(self.peak_rss, rss)
                self.elapsed_seconds = now - self._started_at

            if now - self._last_persisted_at >= self.persist_every:
                self._persist()
                self._last_persisted_at = now

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
        self._last_persisted_at = now

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor,
            name="peak-ram-monitor",
            daemon=True,
        )
        self._thread.start()

        # Create the file immediately.
        self._persist()

    def snapshot(self) -> dict[str, float]:
        rss = self.process.memory_info().rss
        now = time.monotonic()

        with self._lock:
            self.peak_rss = max(self.peak_rss, rss)
            self.elapsed_seconds = now - self._started_at

        self._persist()
        return self._current_metrics()

    def stop(self) -> dict[str, float]:
        if self._thread is None:
            raise RuntimeError("PeakRAMMonitor is not running.")

        self._stop_event.set()
        self._thread.join()

        metrics = self.snapshot()
        self._thread = None
        return metrics
