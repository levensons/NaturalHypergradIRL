import os
import threading
import time
import psutil


class RAM:
    MB = 1024**2

    def __init__(self, interval: float = 0.001):
        if interval <= 0:
            raise ValueError(f"`interval` must be positive, got {interval}.")

        self.interval = interval
        self.process = psutil.Process(os.getpid())

        self.start_rss = 0
        self.peak_rss = 0
        self.end_rss = 0

        self.elapsed_seconds = 0.0
        self.metrics: dict[str, float] | None = None

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
            end_rss = self.end_rss
            elapsed_seconds = self.elapsed_seconds

        return {
            "start_rss_mb": start_rss / self.MB,
            "peak_rss_mb": peak_rss / self.MB,
            "end_rss_mb": end_rss / self.MB,
            "peak_rss_increase_mb": (peak_rss - start_rss) / self.MB,
            "retained_rss_mb": (end_rss - start_rss) / self.MB,
            "released_from_peak_mb": (peak_rss - end_rss) / self.MB,
            "elapsed_seconds": elapsed_seconds,
        }

    def _monitor(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._update()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RAM is already running.")

        rss = self.process.memory_info().rss
        now = time.monotonic()

        with self._lock:
            self.start_rss = rss
            self.peak_rss = rss
            self.end_rss = rss
            self.elapsed_seconds = 0.0

        self._started_at = now
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._monitor, name="ram-monitor", daemon=True)
        self._thread.start()

    def snapshot(self) -> dict[str, float]:
        if self._thread is None:
            raise RuntimeError("RAM monitor is not running.")

        self._update()
        return self._current_metrics()

    def stop(self) -> dict[str, float]:
        if self._thread is None:
            raise RuntimeError("RAM monitor is not running.")

        self._stop_event.set()
        self._thread.join()
        rss = self.process.memory_info().rss

        with self._lock:
            self.peak_rss = max(self.peak_rss, rss)
            self.end_rss = rss
            self.elapsed_seconds = time.monotonic() - self._started_at

        self.metrics = self._current_metrics()
        self._thread = None
        return self.metrics

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    @staticmethod
    def current() -> dict[str, float]:
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        uss = process.memory_full_info().uss
        return {"rss_mb": rss / RAM.MB, "uss_mb": uss / RAM.MB}


class RecordTime:
    def __init__(self, obj=None, attr=None):
        if obj is None and attr is not None:
            raise ValueError("attr cannot be specified without obj.")

        self.obj = obj
        self.attr = attr
        self.elapsed = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start

        if self.obj is not None:
            setattr(self.obj, self.attr, self.elapsed)
    
