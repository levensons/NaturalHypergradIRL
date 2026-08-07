"""
Policy final training.

Usage:
    python -m src.irl.policy_final_training \
        --env lqr \
        --config config/lqr_final_fisher.yaml \
        --checkpoint checkpoints/lqr/fisher.pt

The learned reward is restored from the checkpoint, while the policy is
initialized from scratch and optimized according to `policy_final_training`.

The resulting checkpoint preserves the learned reward and replaces the original
policy with the final trained policy.
"""

import argparse
from pathlib import Path

import mlflow
import torch

from src.algorithms.reinforce import REINFORCE
from src.algorithms.sac import SAC
from src.evaluation.evaluate import SUPPORTED_ENVS, build_env, build_policy, build_reward, import_models_module
from src.evaluation.metrics import inner_loss, learned_reward_stats
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.data import load_trajectories
from src.utils.logging import get_logger
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return


SUPPORTED_AGENTS = {"reinforce", "sac"}


def build_agent(policy, env, final_cfg: dict):
    agent_type = final_cfg["type"]
    params = final_cfg["params"]

    if agent_type == "sac":
        return SAC(
            policy=policy,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=int(params["q_hidden_dim"]),
            n_hidden_layers=int(params["q_n_hidden_layers"]),
            gamma=float(final_cfg["gamma"]),
            alpha=float(final_cfg["alpha"]),
            tau=float(params["tau"]),
            replay_buffer_capacity=int(params["replay_buffer_capacity"]),
        )

    if agent_type == "reinforce":
        return REINFORCE(
            policy=policy,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            gamma=float(final_cfg["gamma"]),
            alpha=float(final_cfg["alpha"]),
        )

    raise ValueError(f"Unsupported agent: {agent_type}. Available: {sorted(SUPPORTED_AGENTS)}")


def train_policy(config: dict, checkpoint: dict, checkpoint_path: str | Path, logger) -> Path:
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
    irl_agent = arch.get("agent")

    env_name = env_cfg["name"]
    models_module, module_path = import_models_module(env_name)

    logger.info(f"Environment: {env_name}")
    logger.info(f"IRL method: {irl_method}")
    logger.info(f"IRL agent: {irl_agent}")
    logger.info(f"Final-training agent: {agent_type}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Models module: {module_path}")

    train_env = build_env(env_cfg, train_env_seed)
    eval_env = build_env(env_cfg, eval_env_seed)

    try:
        reward = build_reward(
            models_module=models_module,
            arch=arch,
        )
        reward.load_state_dict(checkpoint["reward_state_dict"])
        reward.eval()

        logger.info("Learned reward loaded.")

        policy = build_policy(
            models_module=models_module,
            arch=arch,
            env=train_env,
            env_cfg=env_cfg,
        )
        policy.train()

        logger.info("Fresh policy initialized.")

        learned_reward_fn = reward.as_fn()
        train_env.custom_reward_fn = learned_reward_fn

        agent = build_agent(
            policy=policy,
            env=train_env,
            final_cfg=final_cfg,
        )

        if agent_type == "sac":
            random_train_path = Path(data_cfg["random_train_trajs"])
            random_train_trajs = load_trajectories(
                random_train_path,
                map_location="cpu",
            )

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

            l_inner = float(
                inner_loss(
                    agent.policy,
                    reward,
                    trajectories,
                    agent.gamma,
                    agent.alpha,
                )
            )

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
            "params": params,
        }

        torch.save(checkpoint, checkpoint_path)

        logger.info(f"Checkpoint updated with final policy: {checkpoint_path}")

        return Path(checkpoint_path)

    finally:
        train_env.close()
        eval_env.close()


def parse() -> argparse.Namespace:
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
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the source IRL checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = Path(args.config)
    config = load_config(config_path)

    env_name = config["env"]["name"]

    if env_name != args.env:
        raise ValueError(f"Environment mismatch: --env={args.env}, but config contains env.name={env_name}.")

    checkpoint = load_checkpoint(args.checkpoint)

    method = checkpoint["arch"]["method"]
    irl_agent = checkpoint["arch"].get("agent")
    agent_type = config["policy_final_training"]["type"]
    final_cfg = config["policy_final_training"]

    log_cfg = config["logging"]

    logger = get_logger(
        f"policy_final_training_{args.env}",
        log_dir=log_cfg["log_dir"],
    )

    logger.info("=== Final policy training ===")
    logger.info(f"Environment: {args.env}")
    logger.info(f"IRL method: {method}")
    logger.info(f"IRL agent: {irl_agent}")
    logger.info(f"Final-training agent: {agent_type}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    mlflow.set_experiment("policy_final_training")

    with mlflow.start_run(run_name=f"{args.env}_{method}_{agent_type}"):
        mlflow.set_tags(
            {
                "env": args.env,
                "irl_method": method,
                "agent": agent_type,
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
            }
        )

        mlflow.log_params(
            {
                f"agent/{key}": "None" if value is None else value
                for key, value in final_cfg["params"].items()
            }
        )

        train_policy(
            config=config,
            checkpoint=checkpoint,
            checkpoint_path=args.checkpoint,
            logger=logger,
        )


if __name__ == "__main__":
    main()
