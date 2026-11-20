import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.algorithms.fisher_nhd import FisherNHD
from src.irl.builders import import_env_builders
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.resources import RAM, RecordTime
from src.utils.seeding import set_random_seed
from src.utils.torch import load_tensor, num_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialized Fisher solve benchmark.")

    parser.add_argument("--env", required=True, choices=["cartpole", "hopper", "lqr"])
    parser.add_argument("--config", required=True, help="Path to the solver configuration.")
    parser.add_argument("--checkpoint", required=True, help="Path to init.pt containing policy and reward weights.")
    parser.add_argument("--D", dest="d_path", required=True, help="Path to the saved weighted score matrix.")
    parser.add_argument("--g", dest="g_path", required=True, help="Path to the saved outer gradient.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs/benchmarks")
    parser.add_argument("--solution", default=None, help="Optional path to save the inverse-Fisher-vector product.")

    return parser.parse_args()


def build_fisher_optimizer(config: dict, env_builders):
    env_cfg = config["env"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    fisher_cfg = config["fisher"]

    env = env_builders.build_env(env_cfg=env_cfg, seed=int(fisher_cfg["env_seed"]))
    policy = env_builders.build_policy(env=env, policy_cfg=policy_cfg)
    reward = env_builders.build_reward(env=env, reward_cfg=reward_cfg)

    return env, policy, reward


def log_solver_metrics(logger, optimizer: FisherNHD) -> None:
    if optimizer.mode == "cg":
        logger.info("========== CG diagnostics ==========")

        metrics = {
            "iterations": getattr(optimizer, "cg_iterations", None),
            "converged": getattr(optimizer, "cg_converged", None),
            "final_relative_residual": getattr(optimizer, "cg_final_relative_residual", None),
        }

        for key, value in metrics.items():
            if value is not None:
                logger.info(f"{key}: {value}")

    elif optimizer.mode == "sketch":
        logger.info("========== CBSCFD diagnostics ==========")

        metrics = {
            "rows_processed": "sketch_rows_processed",
            "iteration_count": "sketch_iteration_count",
            "extend_count": "sketch_extend_count",
            "svd_count": "sketch_svd_count",
            "extend_time_seconds": "sketch_extend_time",
            "svd_time_seconds": "sketch_svd_time",
            "other_time_seconds": "sketch_other_time",
            "mean_extend_time_seconds": "sketch_mean_extend_time",
            "mean_svd_time_seconds": "sketch_mean_svd_time",
            "mean_other_time_seconds": "sketch_mean_other_time",
            "mean_other_per_iteration_seconds": "sketch_mean_other_per_iteration",
            "svd_fraction": "sketch_svd_fraction",
            "solve_time_seconds": "sketch_solve_time",
            "total_time_seconds": "sketch_total_time",
            "svd_peak_rss_increase_mb": "sketch_svd_peak_rss_increase_mb",
            "qr_peak_rss_increase_mb": "sketch_qr_peak_rss_increase_mb",
            "matmul_peak_rss_increase_mb": "sketch_matmul_peak_rss_increase_mb",
            "sketch_update_peak_rss_increase_mb": "sketch_update_peak_rss_increase_mb",
        }

        for label, attribute in metrics.items():
            value = getattr(optimizer, attribute, None)
            if value is not None:
                logger.info(f"{label}: {value}")


def run_benchmark(
    config: dict,
    checkpoint_path: Path,
    d_path: Path,
    g_path: Path,
    logger,
    solution_path: Path | None = None,
):
    fisher_cfg = config["fisher"]

    if fisher_cfg.get("implementation", "streaming") != "materialized":
        raise ValueError("Benchmark requires implementation='materialized'.")

    mode = fisher_cfg["mode"]

    # --------------------------------------------------
    # 1. LOAD CHECKPOINT
    # --------------------------------------------------

    logger.info("Loading checkpoint...")

    checkpoint = load_checkpoint(checkpoint_path)

    required_keys = {"policy_state_dict", "reward_state_dict"}
    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        raise KeyError(f"Checkpoint is missing keys: {sorted(missing_keys)}")

    env_builders = import_env_builders(config["env"]["name"])
    env, policy, reward = build_fisher_optimizer(config, env_builders)

    device = torch.device("cpu")
    policy.to(device)
    reward.to(device)

    policy.load_state_dict(checkpoint["policy_state_dict"])
    reward.load_state_dict(checkpoint["reward_state_dict"])

    policy.eval()
    reward.eval()

    logger.info(f"Checkpoint loaded: {checkpoint_path}")

    # --------------------------------------------------
    # 2. LOAD D AND G
    # --------------------------------------------------

    logger.info("Loading D and g...")

    D = load_tensor(d_path)
    g = load_tensor(g_path)

    if D.ndim != 2:
        raise ValueError(f"D must be 2-dimensional, got {tuple(D.shape)}.")

    if g.ndim != 1:
        raise ValueError(f"g must be 1-dimensional, got {tuple(g.shape)}.")

    policy_num_params = num_params(policy)

    if D.shape[1] != policy_num_params:
        raise ValueError(f"D has {D.shape[1]} columns, but policy has {policy_num_params} parameters.")

    if g.numel() != policy_num_params:
        raise ValueError(f"g has {g.numel()} elements, but policy has {policy_num_params} parameters.")

    if D.dtype != torch.float32 or g.dtype != torch.float32:
        raise ValueError(f"Materialized solvers expect float32 D and g, got D={D.dtype}, g={g.dtype}.")

    if not torch.isfinite(g).all():
        raise ValueError("g contains NaN or Inf.")

    logger.info(f"D shape: {tuple(D.shape)}")
    logger.info(f"D dtype: {D.dtype}")
    logger.info(f"D size: {D.numel() * D.element_size() / 1024**3:.3f} GB")
    logger.info(f"g shape: {tuple(g.shape)}")
    logger.info(f"g dtype: {g.dtype}")
    logger.info(f"g norm: {g.norm().item():.6e}")
    logger.info(f"Policy parameters: {policy_num_params}")

    # --------------------------------------------------
    # 3. INITIALIZE FISHER NHD
    # --------------------------------------------------

    optimizer = FisherNHD(
        reward=reward,
        policy=policy,
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
        beta=float(fisher_cfg["beta"]),
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        max_grad_norm=fisher_cfg["max_grad_norm"],
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
        implementation="materialized",
        mode=mode,
        cg_max_iters=None if fisher_cfg.get("cg_max_iters") is None else int(fisher_cfg["cg_max_iters"]),
        cg_tol=None if fisher_cfg.get("cg_tol") is None else float(fisher_cfg["cg_tol"]),
        sketch_size=None if fisher_cfg.get("sketch_size") is None else int(fisher_cfg["sketch_size"]),
        sketch_decomposition=str(fisher_cfg.get("sketch_decomposition") or "svd"),
        fisher_batch_size=int(fisher_cfg["fisher_batch_size"]),
    )

    logger.info("========== Benchmark configuration ==========")
    logger.info(f"Mode: {mode}")
    logger.info(f"Fisher regularization: {optimizer.fisher_reg}")

    if mode == "cg":
        logger.info(f"CG max iterations: {optimizer.cg_max_iters}")
        logger.info(f"CG tolerance: {optimizer.cg_tol}")

    elif mode == "sketch":
        logger.info(f"Sketch size: {optimizer.sketch_size}")
        logger.info(f"Decomposition: {optimizer.sketch_decomposition}")

    # --------------------------------------------------
    # 4. FISHER SOLVE BENCHMARK
    # --------------------------------------------------

    logger.info("========== Starting Fisher solve ==========")

    try:
        with RAM() as ram:
            with RecordTime() as timer:
                if mode == "explicit":
                    solution = optimizer.fisher_solve_explicit_fisher_materialized(D, g)

                elif mode == "cg":
                    solution = optimizer.fisher_solve_cg_materialized(D, g, optimizer.cg_max_iters, optimizer.cg_tol)

                elif mode == "sketch":
                    solution = optimizer.fisher_solve_sketch_materialized(D, g, optimizer.sketch_size)

                else:
                    raise ValueError(f"Unknown Fisher solve mode: {mode}.")

    finally:
        env.close()

    # --------------------------------------------------
    # 5. LOG RESOURCES
    # --------------------------------------------------

    logger.info("========== Fisher solve results ==========")
    logger.info(f"Mode: {mode}")
    logger.info(f"Elapsed time: {timer.elapsed:.6f} seconds")

    logger.info("========== RAM metrics ==========")

    for key, value in ram.metrics.items():
        logger.info(f"{key}: {value}")

    log_solver_metrics(logger, optimizer)

    logger.info("========== Benchmark completed ==========")

    return solution


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    d_path = Path(args.d_path)
    g_path = Path(args.g_path)

    config = load_config(config_path)

    if config["env"]["name"] != args.env:
        raise ValueError(f"Environment mismatch: CLI={args.env}, config={config['env']['name']}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.env}_{config['fisher']['mode']}"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp}.log"

    logger = get_logger(f"fisher_solve_{args.env}", log_path=log_path)

    set_random_seed(int(config["fisher"]["random_seed"]))

    logger.info("=== Materialized Fisher solve benchmark ===")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"D: {d_path}")
    logger.info(f"g: {g_path}")
    logger.info(f"Log: {log_path}")

    run_benchmark(
        config=config,
        checkpoint_path=checkpoint_path,
        d_path=d_path,
        g_path=g_path,
        logger=logger,
        solution_path=Path(args.solution) if args.solution is not None else None,
    )


if __name__ == "__main__":
    main()
