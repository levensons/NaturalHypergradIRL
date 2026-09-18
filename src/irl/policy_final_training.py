"""
Policy final training.

Usage:
    python -m src.irl.policy_final_training \
        --env lqr \
        --config config/lqr_final_fisher.yaml \
        --checkpoint checkpoints/lqr/fisher.pt \
        --run-name final_fisher \
        --log-dir logs/lqr/exp1/final_policy_training
"""

import argparse
from pathlib import Path
from datetime import datetime

import mlflow
import torch

from src.evaluation.metrics import inner_loss, learned_reward_stats
from src.irl.builders import SUPPORTED_AGENTS, SUPPORTED_ENVS, build_agent, import_env_builders
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.data import load_trajectories
from src.utils.logging import get_logger
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return


def train_policy(
    config: dict,
    checkpoint: dict,
    checkpoint_path: str | Path,
    env_builders,
    logger,
) -> Path:
    final_cfg = config["policy_final_training"]
    params = final_cfg["params"]
    env_cfg = config["env"]
    data_cfg = config["data"]

    agent_type = final_cfg["type"]

    if agent_type not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported final-training agent: {agent_type}. Available: {sorted(SUPPORTED_AGENTS)}")

    random_seed = int(final_cfg["random_seed"])
    train_env_seed = int(final_cfg["train_env_seed"])
    eval_env_seed = int(final_cfg["eval_env_seed"])

    set_random_seed(random_seed)

    arch = checkpoint["arch"]
    irl_method = arch["method"]

    logger.info(f"IRL method: {irl_method}")
    logger.info(f"Final-training agent: {agent_type}")

    train_env = env_builders.build_env(env_cfg=env_cfg, seed=train_env_seed)
    eval_env = env_builders.build_env(env_cfg=env_cfg, seed=eval_env_seed)

    try:
        reward = env_builders.build_reward(
            env=train_env,
            reward_cfg=config["reward"],
        )
        reward.load_state_dict(checkpoint["reward_state_dict"])
        reward.eval()

        logger.info("Learned reward loaded.")

        policy = env_builders.build_policy(env=train_env, policy_cfg=config["policy"])
        policy.train()

        logger.info("Fresh policy initialized.")

        learned_reward_fn = reward.as_fn()
        train_env.custom_reward_fn = learned_reward_fn

        agent = build_agent(
            policy=policy,
            env=train_env,
            agent_cfg=final_cfg,
            gamma=float(final_cfg["gamma"]),
            alpha=float(final_cfg["alpha"]),
        )

        if agent_type == "sac":
            random_train_path = Path(data_cfg["random_train_trajs"])
            random_train_trajs = load_trajectories(random_train_path, map_location="cpu")

            logger.info(f"Loaded {len(random_train_trajs)} random train trajectories from {random_train_path}")

            agent.replay_buffer.extend_from_trajectories(random_train_trajs)
            agent.replay_buffer.recalc_rewards(learned_reward_fn)

        def validate(ts: int) -> None:
            agent.policy.eval()

            trajectories = collect_trajectories(
                env=eval_env,
                policy=agent.policy,
                n=int(final_cfg["n_agent_eval_trajs"]),
                deterministic=False,
                verbose=False,
            )

            l_inner = float(inner_loss(agent.policy, reward, trajectories, agent.gamma, agent.alpha))
            env_return = float(mean_trajectory_return(trajectories))
            length = float(mean_trajectory_length(trajectories))

            reward_stats = learned_reward_stats(reward, trajectories)
            learned_return = float(reward_stats["return_mean"])
            learned_step_mean = float(reward_stats["step_mean"])

            mlflow.log_metrics(
                {
                    "validation/inner_loss": l_inner,
                    "validation/env_return": env_return,
                    "validation/learned_return": learned_return,
                    "validation/learned_step_mean": learned_step_mean,
                    "validation/length": length,
                },
                step=ts,
            )

            agent.policy.train()

        if agent_type == "sac":
            agent.optimize(
                train_env=train_env,
                total_steps=int(params["total_timesteps"]),
                batch_size=int(params["batch_size"]),
                max_grad_norm=params["max_grad_norm"],
                gradient_update_steps=int(params["gradient_update_steps"]),
                target_update_interval=int(params["target_update_interval"]),
                critic_lr=float(params["critic_lr"]),
                actor_lr=float(params["actor_lr"]),
                validate_fn=validate,
                validate_every=int(final_cfg["validate_every"]),
            )

        elif agent_type == "reinforce":
            agent.optimize(
                train_env=train_env,
                total_steps=int(params["total_steps"]),
                n_traj_per_update=int(params["n_traj_per_update"]),
                max_grad_norm=params["max_grad_norm"],
                actor_lr=float(params["lr_policy"]),
                scheduler_gamma=float(params["scheduler_gamma"]),
                validate_fn=validate,
                validate_every=int(final_cfg["validate_every"]),
            )

        agent.policy.eval()

        checkpoint["policy_state_dict"] = agent.policy.state_dict()
        checkpoint["arch"]["agent"] = agent_type

        checkpoint["final_policy_training"] = {
            "agent": agent_type,
            "random_seed": random_seed,
            "train_env_seed": train_env_seed,
            "eval_env_seed": eval_env_seed,
            "gamma": float(final_cfg["gamma"]),
            "alpha": float(final_cfg["alpha"]),
            "params": dict(params),
        }

        torch.save(checkpoint, checkpoint_path)

        logger.info(f"Checkpoint updated with final policy: {checkpoint_path}")

        return Path(checkpoint_path)

    finally:
        train_env.close()
        eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fresh policy against a learned reward from an IRL checkpoint.",
    )

    parser.add_argument(
        "--env",
        choices=sorted(SUPPORTED_ENVS),
        required=True,
        help="Environment name.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the final-policy training config.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the IRL checkpoint.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name. Defaults to <env>_<method>_<agent>.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where a timestamped training log will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)

    env_name = config["env"]["name"]

    if env_name != args.env:
        raise ValueError(f"Environment mismatch: --env={args.env}, but config contains env.name={env_name}.")

    method = checkpoint["arch"]["method"]
    final_cfg = config["policy_final_training"]
    agent_type = final_cfg["type"]

    run_name = args.run_name or f"final_{method}_{agent_type}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{timestamp}.log"

    logger = get_logger(
        f"policy_final_training_{env_name}",
        log_path=log_path,
    )

    env_builders = import_env_builders(env_name)

    logger.info("=== Final policy training ===")
    logger.info(f"Environment: {env_name}")
    logger.info(f"IRL method: {method}")
    logger.info(f"Final-training agent: {agent_type}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Log path: {log_path}")

    mlflow.set_experiment(env_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "method": method,
                "agent": agent_type,
                "stage": "policy_final_training",
            }
        )

        mlflow.log_params(
            {
                "gamma": final_cfg["gamma"],
                "alpha": final_cfg["alpha"],
                "random_seed": final_cfg["random_seed"],
                "train_env_seed": final_cfg["train_env_seed"],
                "eval_env_seed": final_cfg["eval_env_seed"],
                "validate_every": final_cfg["validate_every"],
                "n_agent_eval_trajs": final_cfg["n_agent_eval_trajs"],
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
            }
        )

        mlflow.log_params(
            {f"agent/{key}": "None" if value is None else value for key, value in final_cfg["params"].items()}
        )

        train_policy(
            config=config,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            env_builders=env_builders,
            logger=logger,
        )


if __name__ == "__main__":
    main()