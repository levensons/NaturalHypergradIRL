"""
ML-IRL with REINFORCE inner agent for CartPole.

Usage:
    python -m src.irl.cartpole.ml_irl
    python -m src.irl.cartpole.ml_irl --config configs/cartpole.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch

from src.utils.env import Environment
from src.irl.cartpole.models import Reward, Policy
from src.algorithms.ml_irl import MLIRL
from src.algorithms.reinforce import REINFORCE
from src.evaluation.metrics import inner_loss, learned_reward_stats, outer_loss, policy_nll, rank_corr
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return
from src.evaluation.video import record_policy_video


def train_ml_irl(config: dict, logger) -> dict:
    ml_irl_cfg = config["ml_irl"]
    inner_cfg = ml_irl_cfg["inner"]
    reinforce_cfg = inner_cfg["params"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    env_cfg = config["env"]
    data_cfg = config["data"]
    ckpt_cfg = config["checkpoint"]

    if inner_cfg["type"] != "reinforce":
        raise ValueError("Expected ml_irl.inner.type = reinforce, " f"got {inner_cfg['type']}.")

    set_random_seed(int(ml_irl_cfg["random_seed"]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    env = Environment(
        id=env_cfg["id"],
        seed=int(ml_irl_cfg["env_seed"]),
        max_episode_steps=int(env_cfg["max_steps"]),
        render_mode="rgb_array",
    )

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    random_train_path = Path(data_cfg["random_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    random_train_trajs = load_trajectories(random_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    logger.info(f"Loaded {len(expert_train_trajs)} expert train trajectories " f"from {expert_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} expert valid trajectories " f"from {expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} random valid trajectories " f"from {random_valid_path}")

    n_outer_steps = int(ml_irl_cfg["n_outer_steps"])
    n_inner_steps = int(ml_irl_cfg["n_inner_steps"])
    n_agent_trajs = int(ml_irl_cfg["n_agent_trajs"])

    reward = Reward(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_hidden_layers=int(reward_cfg["n_hidden_layers"]),
        hidden_dim=int(reward_cfg["hidden_dim"]),
        clamp_magnitude=float(reward_cfg["clamp_magnitude"]),
    ).to(device)

    policy = Policy(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(policy_cfg["hidden_dim"]),
        n_hidden_layers=int(policy_cfg["n_hidden_layers"]),
    ).to(device)

    outer_optimizer = MLIRL(
        reward=reward,
        lr=float(ml_irl_cfg["lr_reward"]),
        gamma=float(ml_irl_cfg["gamma"]),
        alpha=float(ml_irl_cfg["alpha"]),
        max_grad_norm=ml_irl_cfg["max_grad_norm"],
    )

    reinforce = REINFORCE(
        policy=policy,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=float(ml_irl_cfg["gamma"]),
        alpha=float(ml_irl_cfg["alpha"]),
    )

    reinforce_train_env = Environment(
        env_cfg["id"],
        int(inner_cfg["train_env_seed"]),
        int(env_cfg["max_steps"]),
        custom_reward_fn=None,
    )
    reinforce_eval_env = Environment(
        env_cfg["id"],
        int(inner_cfg["eval_env_seed"]),
        int(env_cfg["max_steps"]),
        custom_reward_fn=None,
    )

    def inner_optimize(outer_step: int):
        policy.train()

        current_reward_fn = reward.as_fn()
        reinforce_train_env.custom_reward_fn = current_reward_fn

        def validate(ts):
            policy.eval()

            agent_valid_trajs = collect_trajectories(
                env=reinforce_eval_env,
                policy=policy,
                n=inner_cfg["n_agent_eval_trajs"],
                deterministic=False,
                verbose=False,
            )

            l_inner = inner_loss(reinforce.policy, reward, agent_valid_trajs, reinforce.gamma, reinforce.alpha)
            l_outer_value = outer_loss(reinforce.policy, expert_valid_trajs, reinforce.gamma)

            mlflow.log_metrics(
                {
                    f"reinforce_{outer_step}/l_inner": float(l_inner),
                    f"reinforce_{outer_step}/l_outer": float(l_outer_value),
                    f"reinforce_{outer_step}/env_return": float(mean_trajectory_return(agent_valid_trajs)),
                    f"reinforce_{outer_step}/length": float(mean_trajectory_length(agent_valid_trajs)),
                    f"reinforce_{outer_step}/learned_return": float(learned_reward_stats(reward, agent_valid_trajs)["return_mean"]),
                },
                step=ts,
            )

        reinforce.optimize(
            train_env=reinforce_train_env,
            total_steps=n_inner_steps,
            n_traj_per_update=int(reinforce_cfg["n_traj_per_update"]),
            max_grad_norm=reinforce_cfg["max_grad_norm"],
            actor_lr=float(reinforce_cfg["lr_policy"]),
            scheduler_gamma=float(reinforce_cfg["scheduler_gamma"]),
            # validate_fn=validate,
            validate_every=int(inner_cfg["validate_every"]),
        )

    mlflow.log_params(
        {
            "env_id": env_cfg["id"],
            "gamma": ml_irl_cfg["gamma"],
            "alpha": ml_irl_cfg["alpha"],
            "lr_reward": ml_irl_cfg["lr_reward"],
            "lr_policy": reinforce_cfg["lr_policy"],
            "n_outer_steps": n_outer_steps,
            "n_inner_steps": n_inner_steps,
            "n_agent_trajs": n_agent_trajs,
            "n_traj_per_update": reinforce_cfg["n_traj_per_update"],
            "policy_hidden_dim": policy_cfg["hidden_dim"],
            "reward_hidden_dim": reward_cfg["hidden_dim"],
        }
    )

    arch = {
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "policy_hidden": int(policy_cfg["hidden_dim"]),
        "policy_n_hidden_layers": int(policy_cfg["n_hidden_layers"]),
        "reward_hidden_dim": int(reward_cfg["hidden_dim"]),
        "reward_n_hidden_layers": int(reward_cfg["n_hidden_layers"]),
        "reward_clamp_magnitude": float(reward_cfg["clamp_magnitude"]),
        "method": "ml_irl",
        "agent": "reinforce",
        "env_name": env_cfg["name"],
        "env_id": env_cfg["id"],
        "action_type": env_cfg["action_type"],
    }

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "ml_irl.pt")
    best_env_reward = float("-inf")

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

        l_outer_value = float(outer_loss(policy, expert_valid_trajs, reinforce.gamma))
        l_inner_value = float(inner_loss(policy, reward, agent_trajs, reinforce.gamma, reinforce.alpha))

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
                best_env_reward=best_env_reward,
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

        # record_policy_video(
        #     env,
        #     policy,
        #     video_dir=f"videos/cartpole/ml_irl/",
        #     name_prefix=f"outer_{outer_step}",
        #     deterministic=False,
        #     device=device
        # )

    logger.info(
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

    for outer_step in range(1, n_outer_steps + 1):
        inner_optimize(outer_step)
        agent_train_trajs = collect_trajectories(
            env,
            policy,
            n_agent_trajs,
            deterministic=False,
            desc="agent outer trajs",
            verbose=True,
        )
        log_and_checkpoint(outer_step, agent_train_trajs)

        if outer_step < n_outer_steps:
            # outer_optimizer.sweep_sketch_sizes(
            #     expert_train_trajs,
            #     agent_train_trajs,
            #     [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            #     compare_hypergradients=True
            # )
            outer_optimizer.step(expert_train_trajs, agent_train_trajs)

    env.close()
    reinforce_train_env.close()
    reinforce_eval_env.close()
    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("ML-IRL with REINFORCE — CartPole"))
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("cartpole", args.config)
    config = load_config(config_path)

    log_cfg = config["logging"]

    logger = get_logger("ml_irl_cartpole", log_dir=log_cfg["log_dir"])
    logger.info("=== ML-IRL CartPole REINFORCE ===")

    mlflow.set_experiment("ml_irl")

    with mlflow.start_run(run_name="cartpole"):
        history = train_ml_irl(config, logger)

    report_path = Path(log_cfg["report_dir"]) / "ml_irl_reinforce_cartpole_history.json"

    save_history(history, str(report_path))

    logger.info(f"History saved to {report_path}")


if __name__ == "__main__":
    main()
