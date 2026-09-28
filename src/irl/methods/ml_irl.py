"""
ML-IRL trainer.

Environment-specific objects are constructed through `env_builders`.
The trainer itself is independent of a particular environment.
"""

import os
from pathlib import Path
from types import ModuleType

import mlflow
import torch
import numpy as np

from src.algorithms.ml_irl import MLIRL
from src.evaluation.metrics import inner_loss, learned_reward_stats, outer_loss, policy_nll, rank_corr
from src.irl.builders import SUPPORTED_AGENTS, build_agent, build_arch
from src.utils.checkpoint import save_checkpoint
from src.utils.data import load_trajectories
from src.utils.resources import PeakRAMMonitor, RecordTime
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return


def train_ml_irl(
    config: dict,
    env_builders: ModuleType,
    checkpoint_path: str | Path,
    log_every: int,
    logger,
) -> None:
    ml_irl_cfg = config["ml_irl"]
    inner_cfg = ml_irl_cfg["inner"]
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

    set_random_seed(int(ml_irl_cfg["random_seed"]))

    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    env = env_builders.build_env(
        env_cfg=env_cfg,
        seed=int(ml_irl_cfg["env_seed"]),
    )
    train_env = env_builders.build_env(
        env_cfg=env_cfg,
        seed=int(inner_cfg["train_env_seed"]),
    )
    eval_env = env_builders.build_env(
        env_cfg=env_cfg,
        seed=int(inner_cfg["eval_env_seed"]),
    )

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    random_train_path = Path(data_cfg["random_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    random_train_trajs = load_trajectories(random_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    logger.info(f"Loaded {len(expert_train_trajs)} expert train trajectories from {expert_train_path}")
    logger.info(f"Loaded {len(random_train_trajs)} random train trajectories from {random_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} expert valid trajectories from {expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} random valid trajectories from {random_valid_path}")

    policy = env_builders.build_policy(env=env, policy_cfg=policy_cfg).to(device)
    reward = env_builders.build_reward(env=env, reward_cfg=reward_cfg).to(device)

    outer_optimizer = MLIRL(
        reward=reward,
        alpha=float(ml_irl_cfg["alpha"]),
        gamma=float(ml_irl_cfg["gamma"]),
        lr=float(ml_irl_cfg["lr_reward"]),
        max_grad_norm=ml_irl_cfg["max_grad_norm"],
    )

    agent = build_agent(
        policy=policy,
        env=env,
        agent_cfg=inner_cfg,
        gamma=float(ml_irl_cfg["gamma"]),
        alpha=float(ml_irl_cfg["alpha"]),
    )

    if agent_type == "sac":
        agent.replay_buffer.extend_from_trajectories(random_train_trajs)

    n_outer_steps = int(ml_irl_cfg["n_outer_steps"])
    n_inner_steps = int(ml_irl_cfg["n_inner_steps"])
    n_agent_trajs = int(ml_irl_cfg["n_agent_trajs"])

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

            reward_stats = learned_reward_stats(reward, agent_valid_trajs)

            mlflow.log_metrics(
                {
                    f"{agent_type}_{outer_step}/l_inner": float(l_inner),
                    f"{agent_type}_{outer_step}/l_outer": float(l_outer_value),
                    f"{agent_type}_{outer_step}/env_return": float(mean_trajectory_return(agent_valid_trajs)),
                    f"{agent_type}_{outer_step}/length": float(mean_trajectory_length(agent_valid_trajs)),
                    f"{agent_type}_{outer_step}/learned_return": float(reward_stats["return_mean"]),
                    f"{agent_type}_{outer_step}/learned_step_mean": float(reward_stats["step_mean"]),
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
        method="ml_irl",
        agent_type=agent_type,
    )

    mlflow.log_params(
        {
            "n_outer_steps": n_outer_steps,
            "n_inner_steps": n_inner_steps,
            "n_agent_trajs": n_agent_trajs,
            "alpha": ml_irl_cfg["alpha"],
            "gamma": ml_irl_cfg["gamma"],
            "lr_reward": ml_irl_cfg["lr_reward"],
            "max_grad_norm": ml_irl_cfg["max_grad_norm"],
        }
    )

    mlflow.log_params({f"inner/{key}": "None" if value is None else value for key, value in inner_params.items()})
    mlflow.log_params({f"arch/{key}": value for key, value in arch.items() if not isinstance(value, (list, dict))})

    ram_monitor = PeakRAMMonitor(interval=0.001)
    ram_monitor.start()

    total_optimization_time = 0.0

    inner_step_times = []
    outer_step_times = []
    gradient_times = []

    gradient_start_rss = []
    gradient_peak_rss = []
    gradient_delta_rss = []

    def log_and_checkpoint(outer_step: int, agent_trajs) -> None:
        nonlocal best_l_outer

        ram_metrics = ram_monitor.snapshot()

        l_outer_value = float(outer_loss(policy, expert_valid_trajs, agent.gamma))
        l_inner_value = float(inner_loss(policy, reward, agent_trajs, agent.gamma, agent.alpha))

        agent_length = float(mean_trajectory_length(agent_trajs))
        expert_length = float(mean_trajectory_length(expert_valid_trajs))

        agent_return = float(mean_trajectory_return(agent_trajs))
        expert_return = float(mean_trajectory_return(expert_valid_trajs))

        rank_corr_value = float(rank_corr(reward, expert_valid_trajs + random_valid_trajs))
        policy_nll_value = float(policy_nll(policy, expert_valid_trajs))

        expert_reward_stats = learned_reward_stats(reward, expert_valid_trajs)
        random_reward_stats = learned_reward_stats(reward, random_valid_trajs)

        expert_learned_return = float(expert_reward_stats["return_mean"])
        random_learned_return = float(random_reward_stats["return_mean"])
        expert_step_mean = float(expert_reward_stats["step_mean"])
        random_step_mean = float(random_reward_stats["step_mean"])

        reward_stats = outer_optimizer.stats()

        reward_loss = float(reward_stats["reward_loss"])
        expert_mean_reward = float(reward_stats["expert_learned_return"])
        agent_mean_reward = float(reward_stats["agent_learned_return"])
        raw_grad_norm = float(reward_stats["reward_grad_raw"])
        clipped_grad_norm = float(reward_stats["reward_grad_clipped"])
        lr_reward = float(outer_optimizer.lr)

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
            f"{outer_step:>5} | {l_outer_value:>10.3f} | {l_inner_value:>10.3f} | "
            f"{agent_length:>10.1f} | {agent_return:>10.1f} | {rank_corr_value:>9.3f} | "
            f"{policy_nll_value:>10.3f} | {reward_loss:>10.3f} | {raw_grad_norm:>10.3f} | "
            f"{clipped_grad_norm:>10.3f} | {lr_reward:>12.2e}"
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
                "outer_loss": l_outer_value,
                "inner_loss": l_inner_value,
                "agent_return": agent_return,
                "expert_return": expert_return,
                "agent_length": agent_length,
                "expert_length": expert_length,
                "rank_corr": rank_corr_value,
                "policy_nll": policy_nll_value,
                "reward_loss": reward_loss,
                "reward_grad_norm": raw_grad_norm,
                "reward_grad_norm_clipped": clipped_grad_norm,
                "lr_reward": lr_reward,
                "reward/expert_return": expert_learned_return,
                "reward/random_return": random_learned_return,
                "reward/expert_step_mean": expert_step_mean,
                "reward/random_step_mean": random_step_mean,
                "expert_mean_reward": expert_mean_reward,
                "agent_mean_reward": agent_mean_reward,
                "expert_agent_reward_diff": expert_mean_reward - agent_mean_reward,
                "resources/training_peak_rss_mb": ram_metrics["peak_rss_mb"],
                "resources/training_peak_rss_increase_mb": ram_metrics["peak_rss_increase_mb"],
                "resources/training_elapsed_seconds": ram_metrics["elapsed_seconds"],
            },
            step=outer_step,
        )

    logger.info(
        f"{'Step':>5} | {'L_outer':>10} | {'L_inner':>10} | {'agent_len':>10} | "
        f"{'agent_ret':>10} | {'RankCorr':>9} | {'PolicyNLL':>10} | {'R_loss':>10} | "
        f"{'grad_raw':>10} | {'grad_clip':>10} | {'lr_reward':>12}"
    )

    try:
        for outer_step in range(1, n_outer_steps + 1):
            with RecordTime() as timer:
                inner_optimize(outer_step)

                agent_train_trajs = collect_trajectories(
                    env=env,
                    policy=policy,
                    n=n_agent_trajs,
                    deterministic=False,
                    desc="agent train trajs",
                    verbose=True,
                )

            total_optimization_time += timer.elapsed
            inner_step_times.append(timer.elapsed)

            mlflow.log_metric("timing/inner_step_seconds", float(timer.elapsed), step=outer_step)

            log_and_checkpoint(outer_step, agent_train_trajs)

            if outer_step < n_outer_steps:
                with RecordTime() as timer:
                    outer_optimizer.step(expert_train_trajs, agent_train_trajs)

                total_optimization_time += timer.elapsed
                outer_step_times.append(timer.elapsed)

                gradient_times.append(outer_optimizer.gradient_time)

                gradient_start_rss.append(outer_optimizer.gradient_start_rss_mb)
                gradient_peak_rss.append(outer_optimizer.gradient_peak_rss_mb)
                gradient_delta_rss.append(outer_optimizer.gradient_delta_rss_mb)

                logger.info(
                    "Memory | ML-IRL gradient: "
                    f"start={gradient_start_rss[-1]:.2f} MB, "
                    f"peak={gradient_peak_rss[-1]:.2f} MB, "
                    f"delta={gradient_delta_rss[-1]:.2f} MB"
                )

                logger.info(
                    "Timing | "
                    f"outer step={outer_step_times[-1]:.2f} s | "
                    f"gradient={gradient_times[-1]:.2f} s"
                )

                mlflow.log_metrics(
                    {
                        "timing/outer_step_seconds": float(outer_step_times[-1]),
                        "timing/gradient_seconds": float(gradient_times[-1]),
                        "resources/gradient_start_rss_mb": float(gradient_start_rss[-1]),
                        "resources/gradient_peak_rss_mb": float(gradient_peak_rss[-1]),
                        "resources/gradient_delta_rss_mb": float(gradient_delta_rss[-1]),
                    },
                    step=outer_step,
                )

    finally:
        ram_metrics = ram_monitor.stop()

        inner_step_time_mean = np.mean(inner_step_times)
        inner_step_time_std = np.std(inner_step_times, ddof=1)
        inner_step_time_max = np.max(inner_step_times)

        outer_step_time_mean = np.mean(outer_step_times)
        outer_step_time_std = np.std(outer_step_times, ddof=1)
        outer_step_time_max = np.max(outer_step_times)

        gradient_time_mean = np.mean(gradient_times)
        gradient_time_std = np.std(gradient_times, ddof=1)
        gradient_time_max = np.max(gradient_times)

        gradient_delta_rss_mean = np.mean(gradient_delta_rss)
        gradient_delta_rss_std = np.std(gradient_delta_rss, ddof=1)
        gradient_delta_rss_max = np.max(gradient_delta_rss)
        gradient_peak_rss_max = np.max(gradient_peak_rss)

        logger.info(
            "Training memory | "
            f"start RSS={ram_metrics['start_rss_mb']:.2f} MB | "
            f"peak RSS={ram_metrics['peak_rss_mb']:.2f} MB | "
            f"increase={ram_metrics['peak_rss_increase_mb']:.2f} MB"
        )

        logger.info(
            "Training time | "
            f"total optimization={total_optimization_time:.2f} s"
        )

        logger.info(
            "Inner step time | "
            f"mean={inner_step_time_mean:.2f} s | "
            f"std={inner_step_time_std:.2f} s | "
            f"max={inner_step_time_max:.2f} s"
        )

        logger.info(
            "Outer step time | "
            f"mean={outer_step_time_mean:.2f} s | "
            f"std={outer_step_time_std:.2f} s | "
            f"max={outer_step_time_max:.2f} s"
        )

        logger.info(
            "ML-IRL gradient time | "
            f"mean={gradient_time_mean:.2f} s | "
            f"std={gradient_time_std:.2f} s | "
            f"max={gradient_time_max:.2f} s"
        )

        logger.info(
            "ML-IRL gradient memory | "
            f"delta RSS={gradient_delta_rss_mean:.2f} ± "
            f"{gradient_delta_rss_std:.2f} MB | "
            f"max delta RSS={gradient_delta_rss_max:.2f} MB | "
            f"max peak RSS={gradient_peak_rss_max:.2f} MB"
        )

        try:
            mlflow.log_metrics(
                {
                    "resources/training_peak_rss_mb": ram_metrics["peak_rss_mb"],
                    "resources/training_peak_rss_increase_mb": ram_metrics["peak_rss_increase_mb"],
                    "resources/training_elapsed_seconds": ram_metrics["elapsed_seconds"],
                    "timing/total_optimization_seconds": float(total_optimization_time),
                    "timing/inner_step_seconds_mean": float(inner_step_time_mean),
                    "timing/inner_step_seconds_std": float(inner_step_time_std),
                    "timing/inner_step_seconds_max": float(inner_step_time_max),
                    "timing/outer_step_seconds_mean": float(outer_step_time_mean),
                    "timing/outer_step_seconds_std": float(outer_step_time_std),
                    "timing/outer_step_seconds_max": float(outer_step_time_max),
                    "timing/gradient_seconds_mean": float(gradient_time_mean),
                    "timing/gradient_seconds_std": float(gradient_time_std),
                    "timing/gradient_seconds_max": float(gradient_time_max),
                    "resources/gradient_delta_rss_mb_mean": float(gradient_delta_rss_mean),
                    "resources/gradient_delta_rss_mb_std": float(gradient_delta_rss_std),
                    "resources/gradient_delta_rss_mb_max": float(gradient_delta_rss_max),
                    "resources/gradient_peak_rss_mb_max": float(gradient_peak_rss_max),
                }
            )

        except Exception as error:
            logger.warning(f"Failed to log memory metrics to MLflow: {error}")

    env.close()

    train_env.close()
    eval_env.close()