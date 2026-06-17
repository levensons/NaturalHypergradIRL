"""
    python -m src.prepare_data.collect_trajs --env hopper
    python -m src.prepare_data.collect_trajs --config configs/hopper.yaml

To overwrite existing trajectory files:
    python -m src.prepare_data.collect_trajs --env hopper --overwrite

Before running this script, train the expert first:
    python -m src.prepare_data.train_expert --env hopper
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import Env
import numpy as np
import torch

from src.utils.config import load_config, resolve_config_path
from src.utils.seeding import set_random_seed, set_env_seed
from src.utils.trajectories import collect_trajectories
from src.utils.policies import Policy, RandomPolicy, SB3PolicyWrapper
from src.utils.sb3 import load_sb3_model, normalize_sb3_load_path


def save_trajectories(path: Path, trajectories) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, path)

    lengths = [len(t["states"]) for t in trajectories]
    returns = [float(t["env_rewards"].sum().item()) for t in trajectories]

    print(
        f"Saved {len(trajectories)} trajectories to {path} | "
        f"avg_len={np.mean(lengths):.1f} | "
        f"avg_return={np.mean(returns):.1f}"
    )


def collect_and_save(
    env: Env,
    policy: Policy,
    output_path: Path,
    n_trajectories: int,
    max_steps: int,
    desc: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"Skip existing file: {output_path}")
        return

    trajectories = collect_trajectories(
        env=env,
        policy=policy,
        n=n_trajectories,
        max_steps=max_steps,
        desc=desc,
    )

    save_trajectories(output_path, trajectories)


def collect_all_from_config(config: dict, overwrite: bool) -> None:
    env_cfg = config["env"]
    expert_cfg = config["expert"]

    env_name = env_cfg["name"]
    env_id = env_cfg["id"]
    max_steps = env_cfg["max_steps"]

    split_cfg = config["trajectory_collection"]
    if split_cfg is None:
        raise ValueError("Config must contain `trajectory_collection` section.")

    output_dir = Path(split_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(split_cfg["random_seed"])
    env = gym.make(env_id)
    set_env_seed(env, split_cfg["env_seed"])

    expert_algo = expert_cfg["algo"].lower()
    expert_path = normalize_sb3_load_path(expert_cfg["save_path"])
    expert_model = load_sb3_model(expert_algo, expert_path)
    expert_policy = SB3PolicyWrapper(expert_model)
    random_policy = RandomPolicy(env.action_space)

    print("=" * 80)
    print(f"Collecting trajectories for {env_name}")
    print(f"Env ID: {env_id}")
    print(f"Expert algo: {expert_algo.upper()}")
    print(f"Expert path: {expert_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    saved_files = {}

    expert_split_names = ["train", "valid", "test"]
    random_split_names = ["valid", "test"]

    for split_name in expert_split_names:
        if split_name not in split_cfg:
            continue

        n_trajectories = int(split_cfg[split_name]["n"])

        output_path = output_dir / f"expert_{split_name}_trajs.pt"

        collect_and_save(
            env=env,
            policy=expert_policy,
            output_path=output_path,
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            desc=f"expert {split_name}",
            overwrite=overwrite,
        )

        saved_files[f"expert_{split_name}"] = str(output_path)

    for split_name in random_split_names:
        if split_name not in split_cfg:
            continue

        output_path = output_dir / f"random_{split_name}_trajs.pt"

        collect_and_save(
            env=env,
            policy=random_policy,
            output_path=output_path,
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            desc=f"random {split_name}",
            overwrite=overwrite,
        )

        saved_files[f"random_{split_name}"] = str(output_path)

    metadata = {
        "env_name": env_name,
        "env_id": env_id,
        "max_steps": max_steps,
        "expert_algo": expert_algo,
        "expert_path": str(expert_path),
        "output_dir": str(output_dir),
        "splits": {split_name: split_cfg[split_name] for split_name in expert_split_names if split_name in split_cfg},
        "files": saved_files,
        "trajectory_format": {
            "states": "[T, state_dim], torch.float32",
            "actions": "[T] for discrete or [T, action_dim] for continuous",
            "env_rewards": "[T], torch.float32",
        },
    }

    metadata_path = output_dir / "trajectory_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")

    env.close()


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        choices=["hopper", "cartpole", "pendulum"],
        default=None,
        help="Environment name. Used to load config/<env>.yaml.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit path to config YAML. Overrides --env.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trajectory files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse()
    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)
    collect_all_from_config(config, args.overwrite)


if __name__ == "__main__":
    main()
