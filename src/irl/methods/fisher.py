"""
Fisher IRL trainer.

Environment-specific objects are constructed through `env_builders`.
The trainer itself is independent of a particular environment.
"""

from pathlib import Path
from types import ModuleType

import mlflow
import torch

from src.algorithms.fisher_nhd import FisherNHD
from src.evaluation.metrics import inner_loss, learned_reward_stats, outer_loss, policy_nll, rank_corr
from src.irl.builders import SUPPORTED_AGENTS, build_agent, build_arch
from src.utils.checkpoint import save_checkpoint
from src.utils.data import load_trajectories
from src.utils.resources import PeakRAMMonitor
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return, collect_trajectories_parallel


def train_fisher(
    config: dict,
    env_builders: ModuleType,
    checkpoint_path: str | Path,
    log_every: int,
    mlflow_run_id: str,
    logger,
    n_jobs: int = -1,
) -> None:
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

    # env = env_builders.build_env(env_cfg=env_cfg, seed=int(fisher_cfg["env_seed"]))
    rollout_envs = [
        env_builders.build_env(env_cfg=env_cfg, seed=int(fisher_cfg["env_seed"]) + worker_id)
        for worker_id in range(n_jobs)
    ]
    reference_env = rollout_envs[0]
    
    train_env = env_builders.build_env(env_cfg=env_cfg, seed=int(inner_cfg["train_env_seed"]))
    eval_env = env_builders.build_env(env_cfg=env_cfg, seed=int(inner_cfg["eval_env_seed"]))

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

    policy = env_builders.build_policy(env=reference_env, policy_cfg=policy_cfg).to(device)
    reward = env_builders.build_reward(env=reference_env, reward_cfg=reward_cfg).to(device)
    policy.compile()
    reward.compile()

    outer_optimizer = FisherNHD(
        reward=reward,
        policy=policy,
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        max_grad_norm=fisher_cfg["max_grad_norm"],
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
        use_sketch=bool(fisher_cfg["use_sketch"]),
        sketch_size=fisher_cfg["fisher_sketch_size"],
        fisher_batch_size=int(fisher_cfg["fisher_batch_size"]),
    )

    agent = build_agent(
        policy=policy,
        env=reference_env,
        agent_cfg=inner_cfg,
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
    )

    if agent_type == "sac":
        agent.replay_buffer.extend_from_trajectories(random_train_trajs)

    n_outer_steps = int(fisher_cfg["n_outer_steps"])
    n_inner_steps = int(fisher_cfg["n_inner_steps"])
    n_agent_trajs = int(fisher_cfg["n_agent_trajs"])

    def inner_optimize(outer_step: int) -> None:
        agent.policy.train()

        current_reward_fn = reward.as_fn()
        train_env.custom_reward_fn = current_reward_fn

        if agent_type == "sac":
            agent.replay_buffer.recalc_rewards(current_reward_fn)
            agent.reset_policy()
            agent.reset_critics()
            agent.reset_optimizers()

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
        env=reference_env,
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
            "fisher_sketch_size": fisher_cfg["fisher_sketch_size"],
            "fisher_batch_size": fisher_cfg["fisher_batch_size"],
            "use_sketch": fisher_cfg["use_sketch"],
            "lr_reward": fisher_cfg["lr_reward"],
            "max_grad_norm": fisher_cfg["max_grad_norm"],
            "scheduler_gamma": fisher_cfg["scheduler_gamma"],
        }
    )

    mlflow.log_params(
        {f"inner/{key}": "None" if value is None else value for key, value in inner_params.items()}
    )

    mlflow.log_params(
        {
            f"arch/{key}": value
            for key, value in arch.items()
            if not isinstance(value, (list, dict))
        }
    )

    def log_and_checkpoint(outer_step: int, agent_trajs) -> None:
        nonlocal best_l_outer

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
            },
            step=outer_step,
        )

    logger.info(
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | {'expert_len':>10} | "
        f"{'agent_ret':>10} | {'expert_ret':>10} | {'RankCorr':>9} | {'PolicyNLL':>10} | "
        f"{'hyp_raw':>10} | {'hyp_clip':>10} | {'lr_outer':>12}"
    )

    ram_monitor = PeakRAMMonitor(
        interval=0.05,
        log_every=30.0,
        run_id=mlflow_run_id,
    )
    ram_monitor.start()

    try:
        for outer_step in range(1, n_outer_steps + 1):
            inner_optimize(outer_step)

            agent_train_trajs = collect_trajectories_parallel(
                envs=rollout_envs,
                policy=policy,
                n=n_agent_trajs,
                deterministic=False,
                desc="agent train trajs",
                verbose=True,
            )

            log_and_checkpoint(outer_step, agent_train_trajs)

            if outer_step < n_outer_steps:
                outer_optimizer.step(expert_train_trajs, agent_train_trajs)

    finally:
        ram_metrics = ram_monitor.stop()

        logger.info(
            "Training memory | "
            f"start RSS={ram_metrics['start_rss_mb']:.2f} MB | "
            f"peak RSS={ram_metrics['peak_rss_mb']:.2f} MB | "
            f"increase={ram_metrics['peak_rss_increase_mb']:.2f} MB"
        )

    # env.close()
    for rollout_env in rollout_envs:
        rollout_env.close()

    train_env.close()
    eval_env.close()
