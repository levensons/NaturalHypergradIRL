"""
Fisher IRL trainer.

Environment-specific objects are constructed through `env_builders`.
The trainer itself is independent of a particular environment.
"""

import os
from pathlib import Path
from types import ModuleType

import mlflow
import numpy as np
import torch

from src.algorithms.fisher_nhd import FisherNHD
from src.evaluation.metrics import inner_loss, learned_reward_stats, outer_loss, policy_nll, rank_corr
from src.irl.builders import SUPPORTED_AGENTS, build_agent, build_arch
from src.utils.checkpoint import save_checkpoint
from src.utils.data import load_trajectories
from src.utils.resources import RAM, RecordTime
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return


def train_fisher(
    config: dict,
    env_builders: ModuleType,
    checkpoint_path: str | Path,
    log_every: int,
    logger,
) -> None:

    def log_memory_snapshot(tag: str) -> None:
        memory = RAM.current()

        logger.info(
            f"Memory snapshot | {tag}: "
            f"RSS={memory['rss_mb']:.2f} MB | "
            f"USS={memory['uss_mb']:.2f} MB"
        )

    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]
    inner_params = inner_cfg["params"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    env_cfg = config["env"]
    data_cfg = config["data"]

    agent_type = inner_cfg["type"]

    if agent_type not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported inner agent: {agent_type}. Available: {sorted(SUPPORTED_AGENTS)}")

    if log_every <= 0:
        raise ValueError(f"`log_every` must be positive, got {log_every}.")

    set_random_seed(int(fisher_cfg["random_seed"]))

    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    log_memory_snapshot("before environments")

    env = env_builders.build_env(env_cfg=env_cfg, seed=int(fisher_cfg["env_seed"]))
    
    train_env = env_builders.build_env(env_cfg=env_cfg, seed=int(inner_cfg["train_env_seed"]))
    eval_env = env_builders.build_env(env_cfg=env_cfg, seed=int(inner_cfg["eval_env_seed"]))

    log_memory_snapshot("after environments")

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    random_train_path = Path(data_cfg["random_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    random_train_trajs = load_trajectories(random_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    log_memory_snapshot("after datasets")

    logger.info(f"Loaded {len(expert_train_trajs)} expert train trajectories from {expert_train_path}")
    logger.info(f"Loaded {len(random_train_trajs)} random train trajectories from {random_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} expert valid trajectories from {expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} random valid trajectories from {random_valid_path}")

    policy = env_builders.build_policy(env=env, policy_cfg=policy_cfg).to(device)
    reward = env_builders.build_reward(env=env, reward_cfg=reward_cfg).to(device)

    log_memory_snapshot("after policy and reward")

    outer_optimizer = FisherNHD(
        reward=reward,
        policy=policy,
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        max_grad_norm=fisher_cfg["max_grad_norm"],
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
        mode=str(fisher_cfg["mode"]),
        cg_max_iters=None if fisher_cfg.get("cg_max_iters") is None else int(fisher_cfg["cg_max_iters"]),
        cg_tol=None if fisher_cfg.get("cg_tol") is None else float(fisher_cfg["cg_tol"]),
        sketch_size=None if fisher_cfg.get("sketch_size") is None else int(fisher_cfg["sketch_size"]),
        fisher_batch_size=int(fisher_cfg["fisher_batch_size"]),
    )

    log_memory_snapshot("after FisherNHD")

    agent = build_agent(
        policy=policy,
        env=env,
        agent_cfg=inner_cfg,
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
    )

    log_memory_snapshot("after agent")

    if agent_type == "sac":
        agent.replay_buffer.extend_from_trajectories(random_train_trajs)

    log_memory_snapshot("after replay buffer initialization")

    n_outer_steps = int(fisher_cfg["n_outer_steps"])
    n_inner_steps = int(fisher_cfg["n_inner_steps"])
    n_agent_trajs = int(fisher_cfg["n_agent_trajs"])

    def inner_optimize(outer_step: int) -> None:
        agent.policy.train()

        current_reward_fn = reward.as_fn()
        train_env.custom_reward_fn = current_reward_fn

        if agent_type == "sac":
            agent.replay_buffer.recalc_rewards(current_reward_fn)

            reset_mode = os.getenv("SAC_RESET_MODE", "policy_critics")

            if reset_mode == "nothing":
                agent.reset_optimizers()

            elif reset_mode == "policy":
                agent.reset_policy()
                agent.reset_optimizers()

            elif reset_mode == "critics":
                agent.reset_critics()
                agent.reset_optimizers()

            elif reset_mode == "replay_buffer":
                agent.replay_buffer.reset()

            elif reset_mode == "critics_replay_buffer":
                agent.reset_critics()
                agent.reset_optimizers()
                agent.replay_buffer.reset()

            elif reset_mode == "policy_critics":
                agent.reset_policy()
                agent.reset_critics()
                agent.reset_optimizers()

            elif reset_mode == "policy_replay_buffer":
                agent.reset_policy()
                agent.reset_optimizers()
                agent.replay_buffer.reset()

            elif reset_mode == "policy_critics_replay_buffer":
                agent.reset_policy()
                agent.reset_critics()
                agent.reset_optimizers()
                agent.replay_buffer.reset()

            else:
                raise ValueError(f"Unknown SAC_RESET_MODE: {reset_mode}")

        elif agent_type == "reinforce":
            agent.reset_policy()

        def validate(ts: int) -> None:
            agent.policy.eval()

            agent_valid_trajs = collect_trajectories(
                env=eval_env,
                policy=agent.policy,
                n=int(inner_cfg["n_agent_eval_trajs"]),
                deterministic=False,
                verbose=False,
            )

            l_inner = inner_loss(agent.policy, reward, agent_valid_trajs, agent.gamma, agent.alpha)
            l_outer_value = outer_loss(agent.policy, expert_valid_trajs, agent.gamma)

            l_inner_grad = outer_optimizer.d_inner_d_policy(agent_valid_trajs, verbose=False)
            grad_norm = l_inner_grad.norm()
            grad_rms = grad_norm / (l_inner_grad.numel() ** 0.5)
            grad_abs_max = l_inner_grad.abs().max()

            reward_stats = learned_reward_stats(reward, agent_valid_trajs)

            mlflow.log_metrics(
                {
                    f"{agent_type}_{outer_step}/l_inner": float(l_inner),
                    f"{agent_type}_{outer_step}/l_outer": float(l_outer_value),
                    f"{agent_type}_{outer_step}/env_return": float(mean_trajectory_return(agent_valid_trajs)),
                    f"{agent_type}_{outer_step}/length": float(mean_trajectory_length(agent_valid_trajs)),
                    f"{agent_type}_{outer_step}/learned_return": float(reward_stats["return_mean"]),
                    f"{agent_type}_{outer_step}/learned_step_mean": float(reward_stats["step_mean"]),
                    f"{agent_type}_{outer_step}/inner_grad_norm": grad_norm.item(),
                    f"{agent_type}_{outer_step}/inner_grad_rms": grad_rms.item(),
                    f"{agent_type}_{outer_step}/inner_grad_abs_max": grad_abs_max.item(),
                },
                step=ts,
            )

            agent.policy.train()

        if agent_type == "sac":
            agent.optimize(
                train_env=train_env,
                total_steps=n_inner_steps,
                batch_size=int(inner_params["batch_size"]),
                max_grad_norm=inner_params["max_grad_norm"],
                gradient_update_steps=int(inner_params["gradient_update_steps"]),
                target_update_interval=int(inner_params["target_update_interval"]),
                critic_lr=float(inner_params["critic_lr"]),
                actor_lr=float(inner_params["actor_lr"]),
                validate_fn=validate,
                validate_every=int(inner_cfg["validate_every"]),
            )

        elif agent_type == "reinforce":
            agent.optimize(
                train_env=train_env,
                total_steps=n_inner_steps,
                n_traj_per_update=int(inner_params["n_traj_per_update"]),
                max_grad_norm=inner_params["max_grad_norm"],
                actor_lr=float(inner_params["lr_policy"]),
                scheduler_gamma=float(inner_params["scheduler_gamma"]),
                validate_fn=validate,
                validate_every=int(inner_cfg["validate_every"]),
            )

    best_checkpoint_path = Path(checkpoint_path)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_l_outer = float("inf")

    arch = build_arch(
        env=env,
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
        reward_cfg=reward_cfg,
        method="fisher",
        agent_type=agent_type,
    )

    mlflow.log_params(
        {
            "n_outer_steps": n_outer_steps,
            "n_inner_steps": n_inner_steps,
            "n_agent_trajs": n_agent_trajs,
            "alpha": fisher_cfg["alpha"],
            "gamma": fisher_cfg["gamma"],
            "fisher_reg": fisher_cfg["fisher_reg"],
            "mode": fisher_cfg["mode"],
            "cg_tol": "None" if fisher_cfg.get("cg_tol") is None else fisher_cfg["cg_tol"],
            "cg_max_iters": "None" if fisher_cfg.get("cg_max_iters") is None else fisher_cfg["cg_max_iters"],
            "sketch_size": "None" if fisher_cfg.get("sketch_size") is None else fisher_cfg["sketch_size"],
            "fisher_batch_size": fisher_cfg["fisher_batch_size"],
            "lr_reward": fisher_cfg["lr_reward"],
            "max_grad_norm": fisher_cfg["max_grad_norm"],
            "scheduler_gamma": fisher_cfg["scheduler_gamma"],
        }
    )

    mlflow.log_params({f"inner/{key}": "None" if value is None else value for key, value in inner_params.items()})
    mlflow.log_params({f"arch/{key}": value for key, value in arch.items() if not isinstance(value, (list, dict))})

    log_memory_snapshot("before training")

    ram_monitor = RAM(interval=0.001)
    ram_monitor.start()

    total_optimization_time = 0.0

    inner_step_times = []
    inner_step_peak_rss = []
    inner_step_peak_rss_increase = []
    inner_step_retained_rss = []
    inner_step_released_from_peak = []

    trajectory_collection_times = []
    trajectory_collection_peak_rss = []
    trajectory_collection_peak_rss_increase = []
    trajectory_collection_retained_rss = []
    trajectory_collection_released_from_peak = []

    hypergradient_times = []
    hypergradient_peak_rss = []
    hypergradient_peak_rss_increase = []
    hypergradient_retained_rss = []
    hypergradient_released_from_peak = []

    outer_step_times = []

    outer_grad_times = []
    outer_grad_peak_rss = []
    outer_grad_peak_rss_increase = []
    outer_grad_retained_rss = []
    outer_grad_released_from_peak = []

    fisher_solve_times = []
    fisher_solve_peak_rss = []
    fisher_solve_peak_rss_increase = []
    fisher_solve_retained_rss = []
    fisher_solve_released_from_peak = []

    cross_product_times = []
    cross_product_peak_rss = []
    cross_product_peak_rss_increase = []
    cross_product_retained_rss = []
    cross_product_released_from_peak = []

    def log_and_checkpoint(outer_step: int, agent_trajs) -> None:
        nonlocal best_l_outer

        ram_metrics = ram_monitor.snapshot()

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer_value = outer_loss(policy, expert_valid_trajs, agent.gamma)

        agent_len = mean_trajectory_length(agent_trajs)
        expert_len = mean_trajectory_length(expert_train_trajs)
        agent_ret = mean_trajectory_return(agent_trajs)
        expert_ret = mean_trajectory_return(expert_train_trajs)

        rank_corr_value = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
        policy_nll_value = policy_nll(policy, expert_valid_trajs)

        expert_reward_stats = learned_reward_stats(reward, expert_valid_trajs)
        random_reward_stats = learned_reward_stats(reward, random_valid_trajs)

        expert_learned_return = float(expert_reward_stats["return_mean"])
        random_learned_return = float(random_reward_stats["return_mean"])
        expert_step_mean = float(expert_reward_stats["step_mean"])
        random_step_mean = float(random_reward_stats["step_mean"])

        if l_outer_value < best_l_outer:
            best_l_outer = l_outer_value

            save_checkpoint(
                path=best_checkpoint_path,
                policy=policy,
                reward=reward,
                arch=arch,
                outer_step=outer_step,
                best_l_outer=best_l_outer,
            )

        if outer_step % log_every != 0:
            return

        logger.info(
            f"{outer_step:>5} | {l_outer_value:>10.3f} | {agent_len:>10.1f} | "
            f"{expert_len:>10.1f} | {agent_ret:>10.1f} | {expert_ret:>10.1f} | "
            f"{rank_corr_value:>9.3f} | {policy_nll_value:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )

        logger.info(
            "   [reward] "
            f"expert_ret={expert_learned_return:.3f} "
            f"random_ret={random_learned_return:.3f} "
            f"expert_step={expert_step_mean:.4f} "
            f"random_step={random_step_mean:.4f}"
        )

        mlflow.log_metrics(
            {
                "outer_loss": float(l_outer_value),
                "agent_return": float(agent_ret),
                "expert_return": float(expert_ret),
                "agent_length": float(agent_len),
                "expert_length": float(expert_len),
                "rank_corr": float(rank_corr_value),
                "policy_nll": float(policy_nll_value),
                "hypergrad_norm": float(raw_hypgrad_norm),
                "hypergrad_norm_clipped": float(clipped_hypgrad_norm),
                "lr_outer": float(lr_outer_current),
                "reward/expert_return": expert_learned_return,
                "reward/random_return": random_learned_return,
                "reward/expert_step_mean": expert_step_mean,
                "reward/random_step_mean": random_step_mean,
                "resources/training_start_rss_mb": ram_metrics["start_rss_mb"],
                "resources/training_peak_rss_mb": ram_metrics["peak_rss_mb"],
                "resources/training_end_rss_mb": ram_metrics["end_rss_mb"],
                "resources/training_peak_rss_increase_mb": ram_metrics["peak_rss_increase_mb"],
                "resources/training_retained_rss_mb": ram_metrics["retained_rss_mb"],
                "resources/training_released_from_peak_mb": ram_metrics["released_from_peak_mb"],
                "resources/training_elapsed_seconds": ram_metrics["elapsed_seconds"],
            },
            step=outer_step,
        )

    logger.info(
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | {'expert_len':>10} | "
        f"{'agent_ret':>10} | {'expert_ret':>10} | {'RankCorr':>9} | {'PolicyNLL':>10} | "
        f"{'hyp_raw':>10} | {'hyp_clip':>10} | {'lr_outer':>12}"
    )

    try:
        for outer_step in range(1, n_outer_steps + 1):
            with RAM(interval=0.001) as ram:
                with RecordTime() as timer:
                    inner_optimize(outer_step)

            total_optimization_time += timer.elapsed

            inner_step_times.append(timer.elapsed)
            inner_step_peak_rss.append(ram.metrics["peak_rss_mb"])
            inner_step_peak_rss_increase.append(ram.metrics["peak_rss_increase_mb"])
            inner_step_retained_rss.append(ram.metrics["retained_rss_mb"])
            inner_step_released_from_peak.append(ram.metrics["released_from_peak_mb"])

            logger.info(
                "Memory | inner step: "
                f"start={ram.metrics['start_rss_mb']:.2f} MB, "
                f"peak={ram.metrics['peak_rss_mb']:.2f} MB, "
                f"end={ram.metrics['end_rss_mb']:.2f} MB, "
                f"peak increase={ram.metrics['peak_rss_increase_mb']:.2f} MB, "
                f"retained={ram.metrics['retained_rss_mb']:.2f} MB, "
                f"released={ram.metrics['released_from_peak_mb']:.2f} MB"
            )

            mlflow.log_metrics(
                {
                    "timing/inner_step_seconds": float(timer.elapsed),
                    "resources/inner_step_start_rss_mb": ram.metrics["start_rss_mb"],
                    "resources/inner_step_peak_rss_mb": ram.metrics["peak_rss_mb"],
                    "resources/inner_step_end_rss_mb": ram.metrics["end_rss_mb"],
                    "resources/inner_step_peak_rss_increase_mb": ram.metrics["peak_rss_increase_mb"],
                    "resources/inner_step_retained_rss_mb": ram.metrics["retained_rss_mb"],
                    "resources/inner_step_released_from_peak_mb": ram.metrics["released_from_peak_mb"],
                },
                step=outer_step,
            )

            with RAM(interval=0.001) as ram:
                with RecordTime() as timer:
                    agent_train_trajs = collect_trajectories(
                        env=env,
                        policy=policy,
                        n=n_agent_trajs,
                        deterministic=False,
                        desc="agent train trajs",
                        verbose=True,
                    )

            total_optimization_time += timer.elapsed

            trajectory_collection_times.append(timer.elapsed)
            trajectory_collection_peak_rss.append(ram.metrics["peak_rss_mb"])
            trajectory_collection_peak_rss_increase.append(ram.metrics["peak_rss_increase_mb"])
            trajectory_collection_retained_rss.append(ram.metrics["retained_rss_mb"])
            trajectory_collection_released_from_peak.append(ram.metrics["released_from_peak_mb"])

            logger.info(
                "Memory | trajectory collection: "
                f"start={ram.metrics['start_rss_mb']:.2f} MB, "
                f"peak={ram.metrics['peak_rss_mb']:.2f} MB, "
                f"end={ram.metrics['end_rss_mb']:.2f} MB, "
                f"peak increase={ram.metrics['peak_rss_increase_mb']:.2f} MB, "
                f"retained={ram.metrics['retained_rss_mb']:.2f} MB, "
                f"released={ram.metrics['released_from_peak_mb']:.2f} MB"
            )

            mlflow.log_metrics(
                {
                    "timing/trajectory_collection_seconds": float(timer.elapsed),
                    "resources/trajectory_collection_start_rss_mb": ram.metrics["start_rss_mb"],
                    "resources/trajectory_collection_peak_rss_mb": ram.metrics["peak_rss_mb"],
                    "resources/trajectory_collection_end_rss_mb": ram.metrics["end_rss_mb"],
                    "resources/trajectory_collection_peak_rss_increase_mb": ram.metrics["peak_rss_increase_mb"],
                    "resources/trajectory_collection_retained_rss_mb": ram.metrics["retained_rss_mb"],
                    "resources/trajectory_collection_released_from_peak_mb": ram.metrics["released_from_peak_mb"],
                },
                step=outer_step,
            )

            log_and_checkpoint(outer_step, agent_train_trajs)

            if outer_step < n_outer_steps:
                with RecordTime() as timer:
                    outer_optimizer.step(expert_train_trajs, agent_train_trajs)

                total_optimization_time += timer.elapsed
                outer_step_times.append(timer.elapsed)

                hypergradient_times.append(outer_optimizer.hypergradient_time)
                outer_grad_times.append(outer_optimizer.outer_grad_time)
                fisher_solve_times.append(outer_optimizer.fisher_solve_time)
                cross_product_times.append(outer_optimizer.cross_product_time)

                hypergradient_peak_rss.append(outer_optimizer.hypergradient_peak_rss_mb)
                hypergradient_peak_rss_increase.append(outer_optimizer.hypergradient_peak_rss_increase_mb)
                hypergradient_retained_rss.append(outer_optimizer.hypergradient_retained_rss_mb)
                hypergradient_released_from_peak.append(outer_optimizer.hypergradient_released_from_peak_mb)

                outer_grad_peak_rss.append(outer_optimizer.outer_grad_peak_rss_mb)
                outer_grad_peak_rss_increase.append(outer_optimizer.outer_grad_peak_rss_increase_mb)
                outer_grad_retained_rss.append(outer_optimizer.outer_grad_retained_rss_mb)
                outer_grad_released_from_peak.append(outer_optimizer.outer_grad_released_from_peak_mb)

                fisher_solve_peak_rss.append(outer_optimizer.fisher_solve_peak_rss_mb)
                fisher_solve_peak_rss_increase.append(outer_optimizer.fisher_solve_peak_rss_increase_mb)
                fisher_solve_retained_rss.append(outer_optimizer.fisher_solve_retained_rss_mb)
                fisher_solve_released_from_peak.append(outer_optimizer.fisher_solve_released_from_peak_mb)

                cross_product_peak_rss.append(outer_optimizer.cross_product_peak_rss_mb)
                cross_product_peak_rss_increase.append(outer_optimizer.cross_product_peak_rss_increase_mb)
                cross_product_retained_rss.append(outer_optimizer.cross_product_retained_rss_mb)
                cross_product_released_from_peak.append(outer_optimizer.cross_product_released_from_peak_mb)

                logger.info(
                    "Memory | hypergradient: "
                    f"start={outer_optimizer.hypergradient_start_rss_mb:.2f} MB, "
                    f"peak={outer_optimizer.hypergradient_peak_rss_mb:.2f} MB, "
                    f"end={outer_optimizer.hypergradient_end_rss_mb:.2f} MB, "
                    f"peak increase={outer_optimizer.hypergradient_peak_rss_increase_mb:.2f} MB, "
                    f"retained={outer_optimizer.hypergradient_retained_rss_mb:.2f} MB, "
                    f"released={outer_optimizer.hypergradient_released_from_peak_mb:.2f} MB"
                )

                logger.info(
                    "Memory | outer grad: "
                    f"start={outer_optimizer.outer_grad_start_rss_mb:.2f} MB, "
                    f"peak={outer_optimizer.outer_grad_peak_rss_mb:.2f} MB, "
                    f"end={outer_optimizer.outer_grad_end_rss_mb:.2f} MB, "
                    f"peak increase={outer_optimizer.outer_grad_peak_rss_increase_mb:.2f} MB, "
                    f"retained={outer_optimizer.outer_grad_retained_rss_mb:.2f} MB, "
                    f"released={outer_optimizer.outer_grad_released_from_peak_mb:.2f} MB"
                )

                logger.info(
                    "Memory | Fisher solve: "
                    f"start={outer_optimizer.fisher_solve_start_rss_mb:.2f} MB, "
                    f"peak={outer_optimizer.fisher_solve_peak_rss_mb:.2f} MB, "
                    f"end={outer_optimizer.fisher_solve_end_rss_mb:.2f} MB, "
                    f"peak increase={outer_optimizer.fisher_solve_peak_rss_increase_mb:.2f} MB, "
                    f"retained={outer_optimizer.fisher_solve_retained_rss_mb:.2f} MB, "
                    f"released={outer_optimizer.fisher_solve_released_from_peak_mb:.2f} MB"
                )

                logger.info(
                    "Memory | cross product: "
                    f"start={outer_optimizer.cross_product_start_rss_mb:.2f} MB, "
                    f"peak={outer_optimizer.cross_product_peak_rss_mb:.2f} MB, "
                    f"end={outer_optimizer.cross_product_end_rss_mb:.2f} MB, "
                    f"peak increase={outer_optimizer.cross_product_peak_rss_increase_mb:.2f} MB, "
                    f"retained={outer_optimizer.cross_product_retained_rss_mb:.2f} MB, "
                    f"released={outer_optimizer.cross_product_released_from_peak_mb:.2f} MB"
                )

                logger.info(
                    "Timing | "
                    f"outer step={outer_step_times[-1]:.2f} s | "
                    f"hypergradient={hypergradient_times[-1]:.2f} s | "
                    f"outer grad={outer_grad_times[-1]:.2f} s | "
                    f"Fisher solve={fisher_solve_times[-1]:.2f} s | "
                    f"cross product={cross_product_times[-1]:.2f} s"
                )

                mlflow.log_metrics(
                    {
                        "timing/outer_step_seconds": float(outer_step_times[-1]),
                        "timing/hypergradient_seconds": float(hypergradient_times[-1]),
                        "timing/outer_grad_seconds": float(outer_grad_times[-1]),
                        "timing/fisher_solve_seconds": float(fisher_solve_times[-1]),
                        "timing/cross_product_seconds": float(cross_product_times[-1]),

                        "resources/hypergradient_start_rss_mb": outer_optimizer.hypergradient_start_rss_mb,
                        "resources/hypergradient_peak_rss_mb": outer_optimizer.hypergradient_peak_rss_mb,
                        "resources/hypergradient_end_rss_mb": outer_optimizer.hypergradient_end_rss_mb,
                        "resources/hypergradient_peak_rss_increase_mb": outer_optimizer.hypergradient_peak_rss_increase_mb,
                        "resources/hypergradient_retained_rss_mb": outer_optimizer.hypergradient_retained_rss_mb,
                        "resources/hypergradient_released_from_peak_mb": outer_optimizer.hypergradient_released_from_peak_mb,

                        "resources/outer_grad_start_rss_mb": outer_optimizer.outer_grad_start_rss_mb,
                        "resources/outer_grad_peak_rss_mb": outer_optimizer.outer_grad_peak_rss_mb,
                        "resources/outer_grad_end_rss_mb": outer_optimizer.outer_grad_end_rss_mb,
                        "resources/outer_grad_peak_rss_increase_mb": outer_optimizer.outer_grad_peak_rss_increase_mb,
                        "resources/outer_grad_retained_rss_mb": outer_optimizer.outer_grad_retained_rss_mb,
                        "resources/outer_grad_released_from_peak_mb": outer_optimizer.outer_grad_released_from_peak_mb,

                        "resources/fisher_solve_start_rss_mb": outer_optimizer.fisher_solve_start_rss_mb,
                        "resources/fisher_solve_peak_rss_mb": outer_optimizer.fisher_solve_peak_rss_mb,
                        "resources/fisher_solve_end_rss_mb": outer_optimizer.fisher_solve_end_rss_mb,
                        "resources/fisher_solve_peak_rss_increase_mb": outer_optimizer.fisher_solve_peak_rss_increase_mb,
                        "resources/fisher_solve_retained_rss_mb": outer_optimizer.fisher_solve_retained_rss_mb,
                        "resources/fisher_solve_released_from_peak_mb": outer_optimizer.fisher_solve_released_from_peak_mb,

                        "resources/cross_product_start_rss_mb": outer_optimizer.cross_product_start_rss_mb,
                        "resources/cross_product_peak_rss_mb": outer_optimizer.cross_product_peak_rss_mb,
                        "resources/cross_product_end_rss_mb": outer_optimizer.cross_product_end_rss_mb,
                        "resources/cross_product_peak_rss_increase_mb": outer_optimizer.cross_product_peak_rss_increase_mb,
                        "resources/cross_product_retained_rss_mb": outer_optimizer.cross_product_retained_rss_mb,
                        "resources/cross_product_released_from_peak_mb": outer_optimizer.cross_product_released_from_peak_mb,
                    },
                    step=outer_step,
                )

                if outer_optimizer.mode == "cg":
                    logger.info(
                        "CG diagnostics | "
                        f"iterations={outer_optimizer.cg_iterations} | "
                        f"converged={outer_optimizer.cg_converged} | "
                        f"final_relative_residual={outer_optimizer.cg_final_relative_residual:.3e}"
                    )

                    mlflow.log_metrics(
                        {
                            "cg/iterations": float(outer_optimizer.cg_iterations),
                            "cg/converged": float(outer_optimizer.cg_converged),
                            "cg/final_relative_residual": float(outer_optimizer.cg_final_relative_residual),
                        },
                        step=outer_step,
                    )

    finally:
        ram_metrics = ram_monitor.stop()

        def summarize(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "max": float(np.max(values)),
            }

        inner_step_time = summarize(inner_step_times)
        trajectory_collection_time = summarize(trajectory_collection_times)
        outer_step_time = summarize(outer_step_times)
        hypergradient_time = summarize(hypergradient_times)
        outer_grad_time = summarize(outer_grad_times)
        fisher_solve_time = summarize(fisher_solve_times)
        cross_product_time = summarize(cross_product_times)

        inner_step_peak = summarize(inner_step_peak_rss)
        inner_step_increase = summarize(inner_step_peak_rss_increase)
        inner_step_retained = summarize(inner_step_retained_rss)
        inner_step_released = summarize(inner_step_released_from_peak)

        trajectory_collection_peak = summarize(trajectory_collection_peak_rss)
        trajectory_collection_increase = summarize(trajectory_collection_peak_rss_increase)
        trajectory_collection_retained = summarize(trajectory_collection_retained_rss)
        trajectory_collection_released = summarize(trajectory_collection_released_from_peak)

        hypergradient_peak = summarize(hypergradient_peak_rss)
        hypergradient_increase = summarize(hypergradient_peak_rss_increase)
        hypergradient_retained = summarize(hypergradient_retained_rss)
        hypergradient_released = summarize(hypergradient_released_from_peak)

        outer_grad_peak = summarize(outer_grad_peak_rss)
        outer_grad_increase = summarize(outer_grad_peak_rss_increase)
        outer_grad_retained = summarize(outer_grad_retained_rss)
        outer_grad_released = summarize(outer_grad_released_from_peak)

        fisher_solve_peak = summarize(fisher_solve_peak_rss)
        fisher_solve_increase = summarize(fisher_solve_peak_rss_increase)
        fisher_solve_retained = summarize(fisher_solve_retained_rss)
        fisher_solve_released = summarize(fisher_solve_released_from_peak)

        cross_product_peak = summarize(cross_product_peak_rss)
        cross_product_increase = summarize(cross_product_peak_rss_increase)
        cross_product_retained = summarize(cross_product_retained_rss)
        cross_product_released = summarize(cross_product_released_from_peak)

        def log_time_summary(name: str, stats: dict):
            logger.info(
                f"{name} time | "
                f"mean={stats['mean']:.2f} s | "
                f"std={stats['std']:.2f} s | "
                f"max={stats['max']:.2f} s"
            )

        def log_memory_summary(name: str, peak: dict, increase: dict, retained: dict, released: dict):
            logger.info(
                f"{name} memory | "
                f"peak RSS={peak['mean']:.2f} ± {peak['std']:.2f} MB | "
                f"peak increase={increase['mean']:.2f} ± {increase['std']:.2f} MB | "
                f"retained={retained['mean']:.2f} ± {retained['std']:.2f} MB | "
                f"released={released['mean']:.2f} ± {released['std']:.2f} MB | "
                f"max peak RSS={peak['max']:.2f} MB"
            )

        logger.info(
            "Training memory | "
            f"start RSS={ram_metrics['start_rss_mb']:.2f} MB | "
            f"peak RSS={ram_metrics['peak_rss_mb']:.2f} MB | "
            f"end RSS={ram_metrics['end_rss_mb']:.2f} MB | "
            f"peak increase={ram_metrics['peak_rss_increase_mb']:.2f} MB | "
            f"retained={ram_metrics['retained_rss_mb']:.2f} MB | "
            f"released={ram_metrics['released_from_peak_mb']:.2f} MB"
        )

        logger.info(f"Training time | total optimization={total_optimization_time:.2f} s")

        log_time_summary("Inner step", inner_step_time)
        log_time_summary("Trajectory collection", trajectory_collection_time)
        log_time_summary("Outer step", outer_step_time)
        log_time_summary("Hypergradient", hypergradient_time)
        log_time_summary("Outer grad", outer_grad_time)
        log_time_summary("Fisher solve", fisher_solve_time)
        log_time_summary("Cross product", cross_product_time)

        log_memory_summary("Inner step", inner_step_peak, inner_step_increase, inner_step_retained, inner_step_released)
        log_memory_summary("Trajectory collection", trajectory_collection_peak, trajectory_collection_increase, trajectory_collection_retained, trajectory_collection_released)
        log_memory_summary("Hypergradient", hypergradient_peak, hypergradient_increase, hypergradient_retained, hypergradient_released)
        log_memory_summary("Outer grad", outer_grad_peak, outer_grad_increase, outer_grad_retained, outer_grad_released)
        log_memory_summary("Fisher solve", fisher_solve_peak, fisher_solve_increase, fisher_solve_retained, fisher_solve_released)
        log_memory_summary("Cross product", cross_product_peak, cross_product_increase, cross_product_retained, cross_product_released)

        try:
            mlflow.log_metrics(
                {
                    # Whole training
                    "resources/training_start_rss_mb": ram_metrics["start_rss_mb"],
                    "resources/training_peak_rss_mb": ram_metrics["peak_rss_mb"],
                    "resources/training_end_rss_mb": ram_metrics["end_rss_mb"],
                    "resources/training_peak_rss_increase_mb": ram_metrics["peak_rss_increase_mb"],
                    "resources/training_retained_rss_mb": ram_metrics["retained_rss_mb"],
                    "resources/training_released_from_peak_mb": ram_metrics["released_from_peak_mb"],
                    "resources/training_elapsed_seconds": ram_metrics["elapsed_seconds"],
                    "timing/total_optimization_seconds": float(total_optimization_time),

                    # Inner optimization
                    "timing/inner_step_seconds_mean": inner_step_time["mean"],
                    "timing/inner_step_seconds_std": inner_step_time["std"],
                    "timing/inner_step_seconds_max": inner_step_time["max"],
                    "resources/inner_step_peak_rss_mb_mean": inner_step_peak["mean"],
                    "resources/inner_step_peak_rss_mb_std": inner_step_peak["std"],
                    "resources/inner_step_peak_rss_mb_max": inner_step_peak["max"],
                    "resources/inner_step_peak_rss_increase_mb_mean": inner_step_increase["mean"],
                    "resources/inner_step_peak_rss_increase_mb_std": inner_step_increase["std"],
                    "resources/inner_step_peak_rss_increase_mb_max": inner_step_increase["max"],
                    "resources/inner_step_retained_rss_mb_mean": inner_step_retained["mean"],
                    "resources/inner_step_retained_rss_mb_std": inner_step_retained["std"],
                    "resources/inner_step_retained_rss_mb_max": inner_step_retained["max"],
                    "resources/inner_step_released_from_peak_mb_mean": inner_step_released["mean"],
                    "resources/inner_step_released_from_peak_mb_std": inner_step_released["std"],
                    "resources/inner_step_released_from_peak_mb_max": inner_step_released["max"],

                    # Trajectory collection
                    "timing/trajectory_collection_seconds_mean": trajectory_collection_time["mean"],
                    "timing/trajectory_collection_seconds_std": trajectory_collection_time["std"],
                    "timing/trajectory_collection_seconds_max": trajectory_collection_time["max"],
                    "resources/trajectory_collection_peak_rss_mb_mean": trajectory_collection_peak["mean"],
                    "resources/trajectory_collection_peak_rss_mb_std": trajectory_collection_peak["std"],
                    "resources/trajectory_collection_peak_rss_mb_max": trajectory_collection_peak["max"],
                    "resources/trajectory_collection_peak_rss_increase_mb_mean": trajectory_collection_increase["mean"],
                    "resources/trajectory_collection_peak_rss_increase_mb_std": trajectory_collection_increase["std"],
                    "resources/trajectory_collection_peak_rss_increase_mb_max": trajectory_collection_increase["max"],
                    "resources/trajectory_collection_retained_rss_mb_mean": trajectory_collection_retained["mean"],
                    "resources/trajectory_collection_retained_rss_mb_std": trajectory_collection_retained["std"],
                    "resources/trajectory_collection_retained_rss_mb_max": trajectory_collection_retained["max"],
                    "resources/trajectory_collection_released_from_peak_mb_mean": trajectory_collection_released["mean"],
                    "resources/trajectory_collection_released_from_peak_mb_std": trajectory_collection_released["std"],
                    "resources/trajectory_collection_released_from_peak_mb_max": trajectory_collection_released["max"],

                    # Outer optimization
                    "timing/outer_step_seconds_mean": outer_step_time["mean"],
                    "timing/outer_step_seconds_std": outer_step_time["std"],
                    "timing/outer_step_seconds_max": outer_step_time["max"],

                    # Hypergradient
                    "timing/hypergradient_seconds_mean": hypergradient_time["mean"],
                    "timing/hypergradient_seconds_std": hypergradient_time["std"],
                    "timing/hypergradient_seconds_max": hypergradient_time["max"],
                    "resources/hypergradient_peak_rss_mb_mean": hypergradient_peak["mean"],
                    "resources/hypergradient_peak_rss_mb_std": hypergradient_peak["std"],
                    "resources/hypergradient_peak_rss_mb_max": hypergradient_peak["max"],
                    "resources/hypergradient_peak_rss_increase_mb_mean": hypergradient_increase["mean"],
                    "resources/hypergradient_peak_rss_increase_mb_std": hypergradient_increase["std"],
                    "resources/hypergradient_peak_rss_increase_mb_max": hypergradient_increase["max"],
                    "resources/hypergradient_retained_rss_mb_mean": hypergradient_retained["mean"],
                    "resources/hypergradient_retained_rss_mb_std": hypergradient_retained["std"],
                    "resources/hypergradient_retained_rss_mb_max": hypergradient_retained["max"],
                    "resources/hypergradient_released_from_peak_mb_mean": hypergradient_released["mean"],
                    "resources/hypergradient_released_from_peak_mb_std": hypergradient_released["std"],
                    "resources/hypergradient_released_from_peak_mb_max": hypergradient_released["max"],

                    # Outer gradient
                    "timing/outer_grad_seconds_mean": outer_grad_time["mean"],
                    "timing/outer_grad_seconds_std": outer_grad_time["std"],
                    "timing/outer_grad_seconds_max": outer_grad_time["max"],
                    "resources/outer_grad_peak_rss_mb_mean": outer_grad_peak["mean"],
                    "resources/outer_grad_peak_rss_mb_std": outer_grad_peak["std"],
                    "resources/outer_grad_peak_rss_mb_max": outer_grad_peak["max"],
                    "resources/outer_grad_peak_rss_increase_mb_mean": outer_grad_increase["mean"],
                    "resources/outer_grad_peak_rss_increase_mb_std": outer_grad_increase["std"],
                    "resources/outer_grad_peak_rss_increase_mb_max": outer_grad_increase["max"],
                    "resources/outer_grad_retained_rss_mb_mean": outer_grad_retained["mean"],
                    "resources/outer_grad_retained_rss_mb_std": outer_grad_retained["std"],
                    "resources/outer_grad_retained_rss_mb_max": outer_grad_retained["max"],
                    "resources/outer_grad_released_from_peak_mb_mean": outer_grad_released["mean"],
                    "resources/outer_grad_released_from_peak_mb_std": outer_grad_released["std"],
                    "resources/outer_grad_released_from_peak_mb_max": outer_grad_released["max"],

                    # Fisher solve
                    "timing/fisher_solve_seconds_mean": fisher_solve_time["mean"],
                    "timing/fisher_solve_seconds_std": fisher_solve_time["std"],
                    "timing/fisher_solve_seconds_max": fisher_solve_time["max"],
                    "resources/fisher_solve_peak_rss_mb_mean": fisher_solve_peak["mean"],
                    "resources/fisher_solve_peak_rss_mb_std": fisher_solve_peak["std"],
                    "resources/fisher_solve_peak_rss_mb_max": fisher_solve_peak["max"],
                    "resources/fisher_solve_peak_rss_increase_mb_mean": fisher_solve_increase["mean"],
                    "resources/fisher_solve_peak_rss_increase_mb_std": fisher_solve_increase["std"],
                    "resources/fisher_solve_peak_rss_increase_mb_max": fisher_solve_increase["max"],
                    "resources/fisher_solve_retained_rss_mb_mean": fisher_solve_retained["mean"],
                    "resources/fisher_solve_retained_rss_mb_std": fisher_solve_retained["std"],
                    "resources/fisher_solve_retained_rss_mb_max": fisher_solve_retained["max"],
                    "resources/fisher_solve_released_from_peak_mb_mean": fisher_solve_released["mean"],
                    "resources/fisher_solve_released_from_peak_mb_std": fisher_solve_released["std"],
                    "resources/fisher_solve_released_from_peak_mb_max": fisher_solve_released["max"],

                    # Cross product
                    "timing/cross_product_seconds_mean": cross_product_time["mean"],
                    "timing/cross_product_seconds_std": cross_product_time["std"],
                    "timing/cross_product_seconds_max": cross_product_time["max"],
                    "resources/cross_product_peak_rss_mb_mean": cross_product_peak["mean"],
                    "resources/cross_product_peak_rss_mb_std": cross_product_peak["std"],
                    "resources/cross_product_peak_rss_mb_max": cross_product_peak["max"],
                    "resources/cross_product_peak_rss_increase_mb_mean": cross_product_increase["mean"],
                    "resources/cross_product_peak_rss_increase_mb_std": cross_product_increase["std"],
                    "resources/cross_product_peak_rss_increase_mb_max": cross_product_increase["max"],
                    "resources/cross_product_retained_rss_mb_mean": cross_product_retained["mean"],
                    "resources/cross_product_retained_rss_mb_std": cross_product_retained["std"],
                    "resources/cross_product_retained_rss_mb_max": cross_product_retained["max"],
                    "resources/cross_product_released_from_peak_mb_mean": cross_product_released["mean"],
                    "resources/cross_product_released_from_peak_mb_std": cross_product_released["std"],
                    "resources/cross_product_released_from_peak_mb_max": cross_product_released["max"],
                }
            )

        except Exception as error:
            logger.warning(f"Failed to log final metrics to MLflow: {error}")

    env.close()

    train_env.close()
    eval_env.close()
