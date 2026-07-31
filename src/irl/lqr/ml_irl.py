"""
ML-IRL with SAC inner agent for LQR.

Usage:
    python -m src.irl.lqr.ml_irl
    python -m src.irl.lqr.ml_irl --config configs/lqr.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch

from src.irl.lqr.env import LQR
from src.irl.lqr.models import Reward, Policy
from src.algorithms.ml_irl import MLIRL
from src.algorithms.sac import SAC
from src.evaluation.metrics import inner_loss, learned_reward_stats, outer_loss, policy_nll, rank_corr
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return


def train_ml_irl(config: dict, logger) -> dict:
    ml_irl_cfg = config["ml_irl"]
    inner_cfg = ml_irl_cfg["inner"]
    sac_cfg = inner_cfg["params"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    env_cfg = config["env"]
    data_cfg = config["data"]
    checkpoint_cfg = config["checkpoint"]

    if inner_cfg["type"] != "sac":
        raise ValueError("Expected ml_irl.inner.type = sac, " f"got {inner_cfg['type']}.")

    set_random_seed(int(ml_irl_cfg["random_seed"]))
    env = LQR(
        A=env_cfg["A"],
        B=env_cfg["B"],
        Q=env_cfg["Q"],
        R=env_cfg["R"],
        process_cov=env_cfg["process_cov"],
        initial_cov=env_cfg["initial_cov"],
        max_episode_steps=int(env_cfg["max_steps"]),
        action_limit=float(env_cfg["action_limit"]),
        observation_limit=env_cfg["observation_limit"],
        termination_state_norm=env_cfg["termination_state_norm"],
        reward_scale=float(env_cfg["reward_scale"]),
        seed=int(ml_irl_cfg["env_seed"]),
        custom_reward_fn=None,
    )

    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    random_train_path = Path(data_cfg["random_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    random_train_trajs = load_trajectories(random_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    logger.info(f"Loaded {len(expert_train_trajs)} " f"expert train trajectories from " f"{expert_train_path}")
    logger.info(f"Loaded {len(random_train_trajs)} " f"random train trajectories from " f"{random_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} " f"expert valid trajectories from " f"{expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} " f"random valid trajectories from " f"{random_valid_path}")

    n_outer_steps = int(ml_irl_cfg["n_outer_steps"])
    n_inner_steps = int(ml_irl_cfg["n_inner_steps"])

    n_agent_trajs = int(ml_irl_cfg["n_agent_trajs"])

    policy_hidden_dim = int(policy_cfg["hidden_dim"])
    policy_n_hidden_layers = int(policy_cfg["n_hidden_layers"])

    reward_hidden_dim = int(reward_cfg["hidden_dim"])
    reward_n_hidden_layers = int(reward_cfg["n_hidden_layers"])
    reward_clamp_magnitude = float(reward_cfg["clamp_magnitude"])

    reward = Reward(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_hidden_layers=reward_n_hidden_layers,
        hidden_dim=reward_hidden_dim,
        clamp_magnitude=reward_clamp_magnitude,
    ).to(device)

    policy = Policy(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        hidden_dim=policy_hidden_dim,
        n_hidden_layers=policy_n_hidden_layers,
        log_std_min=float(policy_cfg["log_std_min"]),
        log_std_max=float(policy_cfg["log_std_max"]),
    ).to(device)

    outer_optimizer = MLIRL(
        reward=reward,
        alpha=float(ml_irl_cfg["alpha"]),
        gamma=float(ml_irl_cfg["gamma"]),
        lr=float(ml_irl_cfg["lr_reward"]),
        max_grad_norm=ml_irl_cfg["max_grad_norm"],
    )

    sac = SAC(
        policy=policy,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(sac_cfg["q_hidden_dim"]),
        n_hidden_layers=int(sac_cfg["q_n_hidden_layers"]),
        gamma=float(ml_irl_cfg["gamma"]),
        alpha=float(ml_irl_cfg["alpha"]),
        tau=float(sac_cfg["tau"]),
        replay_buffer_capacity=int(sac_cfg["replay_buffer_capacity"]),
    )

    sac_train_env = LQR(
        A=env_cfg["A"],
        B=env_cfg["B"],
        Q=env_cfg["Q"],
        R=env_cfg["R"],
        process_cov=env_cfg["process_cov"],
        initial_cov=env_cfg["initial_cov"],
        max_episode_steps=int(env_cfg["max_steps"]),
        action_limit=float(env_cfg["action_limit"]),
        observation_limit=env_cfg["observation_limit"],
        termination_state_norm=env_cfg["termination_state_norm"],
        reward_scale=float(env_cfg["reward_scale"]),
        seed=int(inner_cfg["train_env_seed"]),
        custom_reward_fn=None,
    )
    sac_eval_env = LQR(
        A=env_cfg["A"],
        B=env_cfg["B"],
        Q=env_cfg["Q"],
        R=env_cfg["R"],
        process_cov=env_cfg["process_cov"],
        initial_cov=env_cfg["initial_cov"],
        max_episode_steps=int(env_cfg["max_steps"]),
        action_limit=float(env_cfg["action_limit"]),
        observation_limit=env_cfg["observation_limit"],
        termination_state_norm=env_cfg["termination_state_norm"],
        reward_scale=float(env_cfg["reward_scale"]),
        seed=int(inner_cfg["eval_env_seed"]),
        custom_reward_fn=None,
    )

    # sac.replay_buffer.extend_from_trajectories(expert_train_trajs)
    sac.replay_buffer.extend_from_trajectories(random_train_trajs)

    def inner_optimize(outer_step: int) -> None:
        policy.train()

        current_reward_fn = reward.as_fn()
        sac_train_env.custom_reward_fn = current_reward_fn
        sac.replay_buffer.recalc_rewards(current_reward_fn)

        def validate(ts: int) -> None:
            sac.policy.eval()

            agent_valid_trajs = collect_trajectories(
                env=sac_eval_env,
                policy=sac.policy,
                n=inner_cfg["n_agent_eval_trajs"],
                deterministic=False,
                verbose=False,
            )
            
            l_inner = inner_loss(sac.policy, reward, agent_valid_trajs, sac.gamma, sac.alpha)
            l_outer = outer_loss(sac.policy, expert_valid_trajs, sac.gamma)

            mlflow.log_metrics(
                {
                    (f"sac_{outer_step}/l_inner"): float(l_inner),
                    (f"sac_{outer_step}/l_outer"): float(l_outer),
                    (f"sac_{outer_step}/env_return"): float(mean_trajectory_return(agent_valid_trajs)),
                    (f"sac_{outer_step}/length"): float(mean_trajectory_length(agent_valid_trajs)),
                },
                step=ts,
            )

            sac.policy.train()

        sac.optimize(
            train_env=sac_train_env,
            total_steps=n_inner_steps,
            batch_size=int(sac_cfg["batch_size"]),
            max_grad_norm=sac_cfg["max_grad_norm"],
            gradient_update_steps=int(sac_cfg["gradient_update_steps"]),
            target_update_interval=int(sac_cfg["target_update_interval"]),
            critic_lr=float(sac_cfg["critic_lr"]),
            actor_lr=float(sac_cfg["actor_lr"]),
            # validate_fn=validate,
            validate_every=int(inner_cfg["validate_every"]),
        )

    checkpoint_dir = Path(checkpoint_cfg["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(checkpoint_dir / "ml_irl.pt")
    best_env_reward = float("-inf")

    arch = {
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "policy_hidden": (policy_hidden_dim),
        "policy_n_hidden_layers": (policy_n_hidden_layers),
        "reward_hidden": (reward_hidden_dim),
        "reward_n_hidden_layers": (reward_n_hidden_layers),
        "reward_clamp_magnitude": (reward_clamp_magnitude),
        "gamma": float(ml_irl_cfg["gamma"]),
        "alpha": float(ml_irl_cfg["alpha"]),
        "log_std_min": float(policy_cfg["log_std_min"]),
        "log_std_max": float(policy_cfg["log_std_max"]),
        "action_low": (env.action_low.tolist()),
        "action_high": (env.action_high.tolist()),
        "method": "ml_irl",
        "outer_objective": ("horizon_times_agent_minus_expert_mean_point_reward"),
        "agent": "sac",
        "env_name": env_cfg["name"],
        "env_id": env_cfg["id"],
        "action_type": (env_cfg["action_type"]),
    }

    mlflow.log_params(
        {
            "method": "ml_irl",
            "inner_agent": "sac",
            "gamma": ml_irl_cfg["gamma"],
            "alpha": ml_irl_cfg["alpha"],
            "lr_reward": ml_irl_cfg["lr_reward"],
            "n_outer_steps": n_outer_steps,
            "n_inner_steps": n_inner_steps,
            "n_agent_trajs": n_agent_trajs,
            "reward_max_grad_norm": ml_irl_cfg["max_grad_norm"],
            "reward_hidden": reward_hidden_dim,
            "reward_n_hidden_layers": reward_n_hidden_layers,
            "policy_hidden": policy_hidden_dim,
            "policy_n_hidden_layers": (policy_n_hidden_layers),
            "q_hidden_dim": sac_cfg["q_hidden_dim"],
            "q_n_hidden_layers": sac_cfg["q_n_hidden_layers"],
            "critic_lr": sac_cfg["critic_lr"],
            "actor_lr": sac_cfg["actor_lr"],
            "batch_size": sac_cfg["batch_size"],
        }
    )

    history = {
        "l_outer": [],
        "l_inner": [],
        "agent_len": [],
        "expert_len": [],
        "agent_return": [],
        "expert_return": [],
        "rank_corr": [],
        "policy_nll": [],
        "reward_loss": [],
        "expert_mean_reward": [],
        "agent_mean_reward": [],
        "raw_reward_grad_norm": [],
        "clipped_reward_grad_norm": [],
        "lr_reward": [],
        "expert_learned_return": [],
        "random_learned_return": [],
        "expert_learned_step_mean": [],
        "random_learned_step_mean": [],
    }

    def log_and_checkpoint(outer_step: int, agent_trajs) -> None:
        nonlocal best_env_reward

        l_outer_value = float(outer_loss(policy, expert_valid_trajs, sac.gamma))
        l_inner_value = float(inner_loss(policy, reward, agent_trajs, sac.gamma, sac.alpha))

        agent_length = float(mean_trajectory_length(agent_trajs))
        expert_length = float(mean_trajectory_length(expert_valid_trajs))

        agent_return = float(mean_trajectory_return(agent_trajs))
        expert_return = float(mean_trajectory_return(expert_valid_trajs))

        rank_corr_value = float(rank_corr(reward, expert_valid_trajs + random_valid_trajs))

        policy_nll_value = float(policy_nll(policy, expert_valid_trajs))

        expert_reward_stats = learned_reward_stats(reward, expert_valid_trajs)
        random_reward_stats = learned_reward_stats(reward, random_valid_trajs)

        expert_learned_return_valid = float(expert_reward_stats["return_mean"])
        random_learned_return_valid = float(random_reward_stats["return_mean"])

        expert_step_mean = float(expert_reward_stats["step_mean"])
        random_step_mean = float(random_reward_stats["step_mean"])

        reward_stats = outer_optimizer.stats()
        reward_loss = float(reward_stats["reward_loss"])
        expert_mean_reward = float(reward_stats["expert_learned_return"])
        agent_mean_reward = float(reward_stats["agent_learned_return"])
        raw_grad_norm = float(reward_stats["reward_grad_raw"])
        clipped_grad_norm = float(reward_stats["reward_grad_clipped"])

        lr_reward = float(outer_optimizer.lr)

        history["l_outer"].append(l_outer_value)
        history["l_inner"].append(l_inner_value)
        history["agent_len"].append(agent_length)
        history["expert_len"].append(expert_length)
        history["agent_return"].append(agent_return)
        history["expert_return"].append(expert_return)
        history["rank_corr"].append(rank_corr_value)
        history["policy_nll"].append(policy_nll_value)
        history["reward_loss"].append(reward_loss)
        history["expert_mean_reward"].append(expert_mean_reward)
        history["agent_mean_reward"].append(agent_mean_reward)
        history["raw_reward_grad_norm"].append(raw_grad_norm)
        history["clipped_reward_grad_norm"].append(clipped_grad_norm)
        history["lr_reward"].append(lr_reward)
        history["expert_learned_return"].append(expert_learned_return_valid)
        history["random_learned_return"].append(random_learned_return_valid)
        history["expert_learned_step_mean"].append(expert_step_mean)
        history["random_learned_step_mean"].append(random_step_mean)

        if agent_return > best_env_reward:
            best_env_reward = agent_return
            save_checkpoint(
                path=best_checkpoint_path,
                policy=policy,
                reward=reward,
                arch=arch,
                outer_step=outer_step,
                best_env_reward=best_env_reward
            )

        logger.info(
            f"{outer_step:>5} | "
            f"{l_outer_value:>10.3f} | "
            f"{l_inner_value:>10.3f} | "
            f"{agent_length:>10.1f} | "
            f"{agent_return:>10.1f} | "
            f"{rank_corr_value:>9.3f} | "
            f"{policy_nll_value:>10.3f} | "
            f"{reward_loss:>10.3f} | "
            f"{raw_grad_norm:>10.3f} | "
            f"{clipped_grad_norm:>10.3f} | "
            f"{lr_reward:>12.2e}"
        )

        logger.info(
            "   [reward] "
            f"expert_ret={expert_learned_return_valid:.3f} "
            f"random_ret={random_learned_return_valid:.3f} "
            f"expert_step={expert_step_mean:.4f} "
            f"random_step={random_step_mean:.4f} "
        )

        metrics = {
            "outer_loss": l_outer_value,
            "inner_loss": l_inner_value,
            "agent_return": agent_return,
            "expert_return": expert_return,
            "agent_length": agent_length,
            "expert_length": expert_length,
            "rank_corr": rank_corr_value,
            "policy_nll": policy_nll_value,
            "reward_grad_norm": (raw_grad_norm),
            "reward_grad_norm_clipped": (clipped_grad_norm),
            "lr_reward": lr_reward,
            "reward/expert_return": expert_learned_return_valid,
            "reward/random_return": random_learned_return_valid,
            "reward/expert_step_mean": expert_step_mean,
            "reward/random_step_mean": random_step_mean,
            "reward_loss": (reward_loss),
            "expert_mean_reward": (expert_mean_reward),
            "agent_mean_reward": (agent_mean_reward),
            "expert_agent_reward_diff": (expert_mean_reward - agent_mean_reward),
        }
        mlflow.log_metrics(metrics, step=outer_step)

    header = (
        f"{'Step':>5} | "
        f"{'L_outer':>10} | "
        f"{'L_inner':>10} | "
        f"{'agent_len':>10} | "
        f"{'agent_ret':>10} | "
        f"{'RankCorr':>9} | "
        f"{'PolicyNLL':>10} | "
        f"{'R_loss':>10} | "
        f"{'grad_raw':>10} | "
        f"{'grad_clip':>10} | "
        f"{'lr_reward':>12}"
    )
    logger.info(header)

    for outer_step in range(1, n_outer_steps + 1):
        inner_optimize(outer_step=outer_step)
        agent_train_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_trajs,
            deterministic=False,
            desc="agent trajectories",
            verbose=True
        )
        log_and_checkpoint(outer_step, agent_train_trajs)

        if outer_step < n_outer_steps:
            outer_optimizer.step(expert_train_trajs, agent_train_trajs)

    env.close()
    sac_train_env.close()
    sac_eval_env.close()
    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("ML-IRL with SAC — LQR"))
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("lqr", args.config)
    config = load_config(config_path)

    log_cfg = config["logging"]

    logger = get_logger("ml_irl_lqr", log_dir=log_cfg["log_dir"])
    logger.info("=== ML-IRL LQR SAC ===")

    mlflow.set_experiment("ml_irl")

    with mlflow.start_run(run_name="lqr"):
        history = train_ml_irl(config, logger)

    report_path = Path(log_cfg["report_dir"]) / "ml_irl_sac_lqr_history.json"

    save_history(history, str(report_path))

    logger.info(f"History saved to {report_path}")


if __name__ == "__main__":
    main()
