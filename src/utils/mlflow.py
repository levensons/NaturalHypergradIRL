from __future__ import annotations

import os
import signal
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import mlflow


class _MlflowSignalExit(BaseException):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"Received signal {signum}")


def flatten_params(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    params: dict[str, Any] = {}

    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            params.update(flatten_params(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            params[name] = value
        else:
            params[name] = str(value)

    return params


def begin_mlflow_run(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    method: str,
    env_name: str,
    agent: str,
):
    mlflow_cfg = config.get("logging", {}).get("mlflow", {})

    if not mlflow_cfg.get("enabled", True):
        return nullcontext()

    configured_tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    if mlflow_cfg.get("force_tracking_uri", False):
        tracking_uri = configured_tracking_uri
    else:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", configured_tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow_cfg.get("experiment", "irl-bilevel")
    run_name = mlflow_cfg.get("run_name", f"{env_name}-{method}-{agent}")

    artifact_location = mlflow_cfg.get("artifact_location")
    if artifact_location and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment,
            artifact_location=str(artifact_location),
        )
    mlflow.set_experiment(experiment)

    tags = {
        "env": env_name,
        "method": method,
        "agent": agent,
        "config_path": str(config_path),
    }
    tags.update(mlflow_cfg.get("tags", {}))

    run = mlflow.start_run(
        run_name=run_name,
        tags=tags,
        log_system_metrics=bool(mlflow_cfg.get("log_system_metrics", False)),
    )

    mlflow.log_params(flatten_params(config))
    log_artifact_if_exists(config_path, artifact_path="config")

    return _managed_run(run)


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def _end_active_run(status: str, *, signum: int | None = None) -> None:
    if mlflow.active_run() is None:
        return

    if signum is not None:
        mlflow.set_tag("termination_signal", _signal_name(signum))

    mlflow.end_run(status=status)


@contextmanager
def _managed_run(run):
    old_handlers = {}

    def handle_signal(signum, frame):
        raise _MlflowSignalExit(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_signal)

    try:
        yield run
    except _MlflowSignalExit as exc:
        _end_active_run("KILLED", signum=exc.signum)
        raise SystemExit(128 + exc.signum) from None
    except KeyboardInterrupt:
        _end_active_run("KILLED")
        raise
    except BaseException:
        _end_active_run("FAILED")
        raise
    else:
        _end_active_run("FINISHED")
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def log_artifact_if_exists(path: str | Path, artifact_path: str | None = None) -> None:
    path = Path(path)
    if path.exists() and mlflow.active_run() is not None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
