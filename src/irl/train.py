"""
Unified IRL training entrypoint.

Usage:
    python -m src.irl.train \
        --env lqr \
        --method fisher \
        --config config/lqr/exp1.yaml \
        --checkpoint checkpoints/lqr/exp1.pt \
        --run-name exp1 \
        --log-dir logs/lqr/exp1/irl_training \
        --n-jobs 8

A timestamped log file is created inside `--log-dir`.
If `--log-dir` is omitted, logs are written to `logs/`.
"""

import argparse
from pathlib import Path
from datetime import datetime
import os

import mlflow

from src.irl.builders import import_env_builders
from src.irl.trainers.fisher import train_fisher
from src.irl.trainers.ml_irl import train_ml_irl
from src.utils.config import load_config
from src.utils.logging import get_logger


SUPPORTED_ENVS = {"cartpole", "hopper", "lqr"}
SUPPORTED_METHODS = {"fisher", "ml_irl"}

TRAINERS = {
    "fisher": train_fisher,
    "ml_irl": train_ml_irl,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified IRL training.")

    parser.add_argument(
        "--env",
        choices=sorted(SUPPORTED_ENVS),
        required=True,
        help="Environment name.",
    )
    parser.add_argument(
        "--method",
        choices=sorted(SUPPORTED_METHODS),
        required=True,
        help="IRL method.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment config.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path where the best checkpoint will be saved.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name. Defaults to the config filename.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where a timestamped training log will be saved.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log outer-loop metrics every N steps.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel environments used for trajectory collection.",
    )

    args = parser.parse_args()

    if args.log_every <= 0:
        parser.error("--log-every must be positive.")

    if args.n_jobs == -1:
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")

        if slurm_cpus is not None:
            args.n_jobs = int(slurm_cpus)
        else:
            args.n_jobs = os.cpu_count() or 1

    elif args.n_jobs <= 0:
        parser.error("--n-jobs must be positive or -1.")

    return args


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    config = load_config(config_path)

    env_name = config["env"]["name"]

    if env_name != args.env:
        raise ValueError(
            f"Environment mismatch: --env={args.env}, "
            f"but config contains env.name={env_name}."
        )

    if args.method not in config:
        raise ValueError(f"Config does not contain section `{args.method}`.")

    run_name = args.run_name or config_path.stem

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{timestamp}.log"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger(
        f"{args.method}_{env_name}",
        log_path=log_path,
    )

    env_builders = import_env_builders(args.env)
    trainer = TRAINERS[args.method]

    logger.info("=== IRL training ===")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Log path: {log_path}")
    logger.info(f"Log every: {args.log_every}")
    logger.info(f"N jobs: {args.n_jobs}")

    mlflow.set_experiment(args.env)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "env": args.env,
                "method": args.method,
                "agent": config[args.method]["inner"]["type"],
            }
        )

        mlflow.log_params(
            {
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "log_every": args.log_every,
                "n_jobs": args.n_jobs,
            }
        )

        trainer(
            config=config,
            env_builders=env_builders,
            checkpoint_path=checkpoint_path,
            log_every=args.log_every,
            mlflow_run_id=run.info.run_id,
            logger=logger,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
