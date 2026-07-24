"""Neural Fisher-NHD with persistent SAC on bounded LQR.

This entry point registers the LQR environment and then deliberately delegates
to the Hopper Fisher trainer. Consequently the benchmark exercises the same
neural reward, squashed-Gaussian policy, critics, replay buffer, persistent SAC,
empirical Fisher, CG solve, and scalar cross-product implementation.

Generate the expert data first, then run::

    python -m src.irl.lqr.sac_oracle --config config/neural_lqr.yaml
    python -m src.irl.lqr.fisher --config config/neural_lqr.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

import mlflow

from src.irl.hopper.fisher import train_bilevel
from src.irl.lqr.env import register_lqr_env
from src.utils.config import load_config
from src.utils.logging import get_logger, save_history
from src.utils.mlflow import begin_mlflow_run, log_artifact_if_exists


_MLFLOW_METRICS = (
    "validation/agent_return",
    "validation/best_agent_return",
    "validation/expert_return",
    "outer/loss",
    "reward/rank_corr",
    "gradient/fisher_ml_cosine",
)


def _minimal_mlflow_metrics(record: dict) -> dict[str, float]:
    """Keep the neural-LQR MLflow surface intentionally small."""
    metrics = {}
    for name in _MLFLOW_METRICS:
        value = record.get(name)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            metrics[name] = value
    return metrics


def _history_record(history: dict, step: int, best_return: float) -> dict:
    return {
        "validation/agent_return": history["agent_return"][step],
        "validation/best_agent_return": best_return,
        "validation/expert_return": history["expert_return"][step],
        "outer/loss": history["l_outer"][step],
        "reward/rank_corr": history["rank_corr"][step],
        "gradient/fisher_ml_cosine": history["fisher_ml_cosine"][step],
    }


def _save_existing_best_checkpoint(
    config: dict,
    history: dict,
    checkpoint_run_id: str | None,
) -> tuple[Path, int, float]:
    returns = [float(value) for value in history["agent_return"]]
    best_step = max(range(len(returns)), key=returns.__getitem__)
    best_return = returns[best_step]
    checkpoint_dir = Path(config["checkpoint"]["dir"])
    suffix = f"_outer_{best_step:04d}.pt"

    if checkpoint_run_id is not None:
        source = checkpoint_dir / f"fisher_sac_{checkpoint_run_id}{suffix}"
        if not source.is_file():
            raise FileNotFoundError(f"Best source checkpoint does not exist: {source}")
    else:
        candidates = list(checkpoint_dir.glob(f"fisher_sac_*{suffix}"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint for best outer step {best_step} under {checkpoint_dir}."
            )
        source = max(candidates, key=lambda path: path.stat().st_mtime_ns)

    prefix = source.name[: -len(suffix)]
    destination = checkpoint_dir / f"{prefix}_best.pt"
    shutil.copy2(source, destination)
    return destination, best_step, best_return


def _backfill_existing_run(
    config: dict,
    config_path: str,
    history_path: Path,
    checkpoint_run_id: str | None,
    logger,
) -> None:
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    n_steps = len(history.get("agent_return", []))
    if n_steps == 0:
        raise ValueError(f"History contains no agent returns: {history_path}")

    best_checkpoint, best_step, best_return = _save_existing_best_checkpoint(
        config, history, checkpoint_run_id
    )
    with begin_mlflow_run(
        config,
        config_path,
        method="fisher",
        env_name="neural_lqr",
        agent="sac",
    ):
        if mlflow.active_run() is not None:
            mlflow.set_tags(
                {
                    "backfilled": "true",
                    "source_history": str(history_path),
                    "checkpoint_run_id": checkpoint_run_id or "auto",
                    "validation_source": "outer_rollouts_backfill",
                    "best_checkpoint": best_checkpoint.name,
                    "best_validation_step": str(best_step),
                }
            )
            running_best = float("-inf")
            for step in range(n_steps):
                running_best = max(
                    running_best, float(history["agent_return"][step])
                )
                mlflow.log_metrics(
                    _minimal_mlflow_metrics(
                        _history_record(history, step, running_best)
                    ),
                    step=step,
                )
            log_artifact_if_exists(history_path, artifact_path="reports")
            log_artifact_if_exists(best_checkpoint, artifact_path="checkpoints")

    logger.info(
        "Backfilled neural-LQR run to MLflow | "
        f"steps={n_steps} | best_step={best_step} | "
        f"best_return={best_return:.3f} | checkpoint={best_checkpoint}"
    )


def _check_data(config: dict) -> None:
    required = (
        "expert_train_trajs",
        "expert_valid_trajs",
        "random_valid_trajs",
    )
    missing = [
        config["data"][name]
        for name in required
        if not Path(config["data"][name]).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Neural LQR demonstration data is missing:\n"
            f"{formatted}\n"
            "Generate it with `python -m src.irl.lqr.sac_oracle "
            "--config config/neural_lqr.yaml`."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neural_lqr.yaml")
    parser.add_argument(
        "--backfill-history",
        default=None,
        help="Log an existing Fisher history to MLflow instead of training.",
    )
    parser.add_argument(
        "--checkpoint-run-id",
        default=None,
        help="Checkpoint identifier for backfill, for example job_4181729.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log_cfg = config["logging"]
    logger = get_logger(
        f"fisher_{config['env']['name']}", log_dir=log_cfg["log_dir"]
    )

    if args.backfill_history is not None:
        _backfill_existing_run(
            config,
            args.config,
            Path(args.backfill_history),
            args.checkpoint_run_id,
            logger,
        )
        return

    register_lqr_env(config)
    _check_data(config)

    reward_max_grad_norm = os.environ.get("REWARD_MAX_GRAD_NORM")
    if reward_max_grad_norm is not None:
        value = float(reward_max_grad_norm)
        if value <= 0.0:
            raise ValueError("REWARD_MAX_GRAD_NORM must be positive.")
        config["fisher"]["max_grad_norm"] = value

    logger.info("=== Fisher-NHD neural SAC on bounded LQR ===")
    logger.info(
        "Reusing Hopper trainer | "
        f"policy={config['policy']['n_hidden_layers']}x{config['policy']['hidden_dim']} | "
        f"reward={config['reward']['n_hidden_layers']}x{config['reward']['hidden_dim']} | "
        f"inner_steps={config['fisher']['n_inner_steps']}"
    )

    best_checkpoint_path = None

    def log_selected_metrics(step: int, record: dict) -> None:
        nonlocal best_checkpoint_path
        if mlflow.active_run() is not None:
            mlflow.log_metrics(_minimal_mlflow_metrics(record), step=step)
        if record.get("best_checkpoint_path") is not None:
            best_checkpoint_path = Path(record["best_checkpoint_path"])

    with begin_mlflow_run(
        config,
        args.config,
        method="fisher",
        env_name="neural_lqr",
        agent="sac",
    ):
        if mlflow.active_run() is not None:
            mlflow.set_tag("validation_source", "sac_eval_environment")
        history = train_bilevel(
            config,
            logger,
            enable_mlflow=False,
            metrics_callback=log_selected_metrics,
            save_best_checkpoint=True,
        )
        report_path = Path(log_cfg["report_dir"]) / "fisher_sac_history.json"
        save_history(history, str(report_path))
        log_artifact_if_exists(report_path, artifact_path="reports")
        if best_checkpoint_path is not None and mlflow.active_run() is not None:
            mlflow.set_tag("best_checkpoint", best_checkpoint_path.name)
            log_artifact_if_exists(best_checkpoint_path, artifact_path="checkpoints")
        logger.info(f"Saved Fisher history to {report_path}")


if __name__ == "__main__":
    main()
