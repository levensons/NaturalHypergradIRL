"""Train the neural SAC expert and create data for the bounded LQR benchmark.

Usage::

    python -m src.irl.lqr.sac_oracle --config config/neural_lqr.yaml

The expert uses exactly the same squashed-Gaussian policy and custom SAC class
as the Fisher benchmark. Demonstrations are sampled stochastically because the
inner objective has a fixed entropy temperature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from src.agents.sac import SAC, observation_normalization_kwargs
from src.irl.hopper.fisher import Policy
from src.irl.lqr.env import register_lqr_env
from src.utils.config import load_config
from src.utils.env import Environment
from src.utils.logging import get_logger, save_history
from src.utils.policies import RandomPolicy
from src.utils.seeding import set_random_seed
from src.utils.trajectories import (
    collect_trajectories,
    mean_trajectory_length,
    mean_trajectory_return,
)


def build_policy_and_sac(config: dict, device: torch.device):
    env_cfg = config["env"]
    fisher_cfg = config["fisher"]
    sac_cfg = fisher_cfg["inner"]["sac"]
    policy_cfg = config["policy"]
    probe_env = Environment(env_cfg["id"], int(config["expert"]["env_seed"]))
    policy = Policy(
        state_dim=probe_env.state_dim,
        action_dim=probe_env.action_dim,
        action_low=probe_env.action_low,
        action_high=probe_env.action_high,
        hidden_dim=int(policy_cfg["hidden_dim"]),
        n_hidden_layers=int(policy_cfg["n_hidden_layers"]),
        log_std_min=float(policy_cfg["log_std_min"]),
        log_std_max=float(policy_cfg["log_std_max"]),
        activation=str(policy_cfg.get("activation", "relu")),
        **observation_normalization_kwargs(sac_cfg),
    ).to(device)
    sac = SAC(
        policy=policy,
        state_dim=probe_env.state_dim,
        action_dim=probe_env.action_dim,
        hidden_dim=int(sac_cfg["q_hidden_dim"]),
        n_hidden_layers=int(sac_cfg["q_n_hidden_layers"]),
        gamma=float(fisher_cfg["discount"]),
        alpha=float(fisher_cfg["alpha"]),
        tau=float(sac_cfg["tau"]),
        replay_buffer_capacity=int(sac_cfg["replay_buffer_capacity"]),
    )
    dimensions = (probe_env.state_dim, probe_env.action_dim)
    probe_env.close()
    return policy, sac, dimensions


def save_expert_checkpoint(
    config: dict,
    policy: Policy,
    sac: SAC,
    *,
    total_steps: int,
) -> Path:
    path = Path(config["expert"]["save_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "q1_state_dict": sac.q1.state_dict(),
            "q2_state_dict": sac.q2.state_dict(),
            "total_env_steps": sac.total_env_steps,
            "total_gradient_updates": sac.total_gradient_updates,
            "requested_total_steps": total_steps,
            "policy": dict(config["policy"]),
            "env_id": config["env"]["id"],
            "lqr": dict(config["lqr"]),
        },
        path,
    )
    return path


def load_expert_policy(config: dict, device: torch.device) -> Policy:
    policy, _, _ = build_policy_and_sac(config, device)
    path = Path(config["expert"]["save_path"])
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def _canonicalize_trajectories(trajectories) -> None:
    """Match the tensor trajectory format consumed by Hopper metrics."""
    for trajectory in trajectories:
        trajectory["states"] = torch.as_tensor(
            trajectory["states"], dtype=torch.float32
        )
        trajectory["actions"] = torch.as_tensor(
            trajectory["actions"], dtype=torch.float32
        )
        trajectory["env_rewards"] = torch.as_tensor(
            trajectory["env_rewards"], dtype=torch.float32
        )


def collect_datasets(config: dict, expert_policy: Policy, logger) -> dict[str, str]:
    env_cfg = config["env"]
    collection_cfg = config["trajectory_collection"]
    output_dir = Path(collection_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(env_cfg["max_steps"])
    base_seed = int(collection_cfg["env_seed"])
    files: dict[str, str] = {}

    split_names = ("train", "valid", "test")
    for index, split in enumerate(split_names):
        count = int(collection_cfg[split]["n"])
        env = Environment(env_cfg["id"], base_seed + index)
        try:
            trajectories = collect_trajectories(
                env=env,
                policy=expert_policy,
                n=count,
                max_steps=max_steps,
                desc=f"LQR expert {split}",
            )
        finally:
            env.close()
        _canonicalize_trajectories(trajectories)
        path = output_dir / f"expert_{split}_trajs.pt"
        torch.save(trajectories, path)
        files[f"expert_{split}"] = str(path)
        logger.info(
            f"Saved {count} expert {split} trajectories to {path} | "
            f"return={mean_trajectory_return(trajectories):.3f} | "
            f"length={mean_trajectory_length(trajectories):.1f}"
        )

    for index, split in enumerate(("valid", "test")):
        count = int(collection_cfg[split]["n"])
        env = Environment(env_cfg["id"], base_seed + 100 + index)
        random_policy = RandomPolicy(env.action_space)
        try:
            trajectories = collect_trajectories(
                env=env,
                policy=random_policy,
                n=count,
                max_steps=max_steps,
                desc=f"LQR random {split}",
            )
        finally:
            env.close()
        _canonicalize_trajectories(trajectories)
        path = output_dir / f"random_{split}_trajs.pt"
        torch.save(trajectories, path)
        files[f"random_{split}"] = str(path)
        logger.info(
            f"Saved {count} random {split} trajectories to {path} | "
            f"return={mean_trajectory_return(trajectories):.3f} | "
            f"length={mean_trajectory_length(trajectories):.1f}"
        )

    metadata = {
        "env_name": env_cfg["name"],
        "env_id": env_cfg["id"],
        "expert_checkpoint": config["expert"]["save_path"],
        "stochastic_expert_actions": True,
        "files": files,
        "lqr": config["lqr"],
    }
    metadata_path = output_dir / "trajectory_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files["metadata"] = str(metadata_path)
    return files


def train_oracle(config: dict, logger, total_steps_override: int | None = None):
    register_lqr_env(config)
    expert_cfg = config["expert"]
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]
    sac_cfg = inner_cfg["sac"]
    env_cfg = config["env"]
    set_random_seed(int(expert_cfg["random_seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, sac, _ = build_policy_and_sac(config, device)
    train_env = Environment(env_cfg["id"], int(inner_cfg["train_env_seed"]))
    eval_env = Environment(env_cfg["id"], int(inner_cfg["eval_env_seed"]))
    total_steps = (
        int(total_steps_override)
        if total_steps_override is not None
        else int(expert_cfg["total_timesteps"])
    )
    validate_every = int(sac_cfg["validate_every"])
    n_eval_traj = int(sac_cfg.get("n_eval_traj", 20))
    history = {"env_step": [], "return": [], "length": [], "best_return": []}
    best_return = float("-inf")
    best_policy_state = None

    def validate(ts: int) -> None:
        nonlocal best_return, best_policy_state
        policy.eval()
        trajectories = collect_trajectories(
            env=eval_env,
            policy=policy,
            n=n_eval_traj,
            max_steps=int(env_cfg["max_steps"]),
            verbose=False,
        )
        agent_return = mean_trajectory_return(trajectories)
        agent_length = mean_trajectory_length(trajectories)
        history["env_step"].append(sac.total_env_steps)
        history["return"].append(agent_return)
        history["length"].append(agent_length)
        if agent_return > best_return:
            best_return = agent_return
            best_policy_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }
        history["best_return"].append(best_return)
        logger.info(
            f"Oracle validation | env_steps={sac.total_env_steps} | "
            f"phase_step={ts} | return={agent_return:.3f} | "
            f"length={agent_length:.1f}"
        )
        policy.train()

    started = perf_counter()
    try:
        validate(0)
        sac.optimize(
            train_env=train_env,
            total_steps=total_steps,
            learning_starts=int(sac_cfg["learning_starts"]),
            batch_size=int(sac_cfg["batch_size"]),
            max_grad_norm=sac_cfg["max_grad_norm"],
            gradient_update_steps=int(sac_cfg["gradient_update_steps"]),
            target_update_interval=int(sac_cfg["target_update_interval"]),
            critic_lr=float(sac_cfg["critic_lr"]),
            actor_lr=float(sac_cfg["actor_lr"]),
            reward_fn=None,
            validate_fn=validate,
            validate_every=validate_every,
        )
        if not history["env_step"] or history["env_step"][-1] != sac.total_env_steps:
            validate(total_steps)
    finally:
        train_env.close()
        eval_env.close()
    elapsed = perf_counter() - started
    if best_policy_state is None:
        raise RuntimeError("Oracle validation did not produce a policy checkpoint.")
    policy.load_state_dict(best_policy_state)
    policy.eval()
    history["training_seconds"] = elapsed
    logger.info(
        f"Oracle SAC training complete in {elapsed:.3f}s | "
        f"restored_best_return={best_return:.3f}"
    )
    return policy, sac, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neural_lqr.yaml")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--collect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    register_lqr_env(config)
    logger = get_logger(
        f"oracle_sac_{config['env']['name']}",
        log_dir=config["logging"]["log_dir"],
    )

    if args.collect_only:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy = load_expert_policy(config, device)
        history = {
            "env_step": [],
            "return": [],
            "length": [],
            "best_return": [],
        }
    else:
        policy, sac, history = train_oracle(
            config, logger, total_steps_override=args.total_steps
        )
        checkpoint_path = save_expert_checkpoint(
            config,
            policy,
            sac,
            total_steps=(
                args.total_steps
                if args.total_steps is not None
                else int(config["expert"]["total_timesteps"])
            ),
        )
        logger.info(f"Saved oracle checkpoint to {checkpoint_path}")

    collect_datasets(config, policy, logger)
    report_path = Path(config["logging"]["report_dir"]) / "oracle_sac_history.json"
    save_history(history, str(report_path))
    logger.info(f"Saved oracle history to {report_path}")


if __name__ == "__main__":
    main()
