import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO, SAC
from tqdm import tqdm


def load_config(path: str) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(env_name: str | None, config_path: str | None) -> str:
    if config_path is not None:
        return config_path

    if env_name is None:
        raise ValueError("Specify either --env or --config.")

    return f"config/{env_name}.yaml"


def seed_env(env, seed: int):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


class SB3PolicyWrapper:
    def __init__(self, model, action_space):
        self.model = model
        self.action_space = action_space

    def sample_action(self, state):
        action, _ = self.model.predict(state, deterministic=True)

        if hasattr(self.action_space, "low"):
            action = np.clip(action, self.action_space.low, self.action_space.high)
            return np.asarray(action, dtype=np.float32)

        return int(action)

    def eval(self):
        pass


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def sample_action(self, state):
        action = self.action_space.sample()

        if np.isscalar(action) or np.asarray(action).shape == ():
            return int(action)

        return np.asarray(action, dtype=np.float32)

    def eval(self):
        pass


def load_expert_model(algo: str, expert_path: Path):
    algo = algo.lower()

    if algo == "sac":
        return SAC.load(str(expert_path))

    if algo == "ppo":
        return PPO.load(str(expert_path))

    raise ValueError(f"Unknown expert algorithm: {algo}")


def collect_trajectories(
    env,
    policy,
    n_trajectories: int,
    max_steps: int,
    seed_start: int,
    desc: str,
):
    trajectories = []

    for i in tqdm(range(n_trajectories), desc=desc, leave=False):
        traj_seed = seed_start + i

        state, _ = env.reset(seed=traj_seed)
        env.action_space.seed(traj_seed)
        env.observation_space.seed(traj_seed)

        states = []
        actions = []
        env_rewards = []
        dones = []

        for _ in range(max_steps):
            action = policy.sample_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(torch.as_tensor(state, dtype=torch.float32))

            if np.isscalar(action) or np.asarray(action).shape == ():
                actions.append(torch.tensor(action, dtype=torch.long))
            else:
                actions.append(torch.as_tensor(action, dtype=torch.float32))

            env_rewards.append(float(reward))
            dones.append(float(done))

            state = next_state

            if done:
                break

        trajectory = {
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "env_rewards": torch.tensor(env_rewards, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
            "seed": traj_seed,
        }

        trajectories.append(trajectory)

    return trajectories


def save_trajectories(path: Path, trajectories):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, path)

    lengths = [len(t["states"]) for t in trajectories]
    returns = [float(t["env_rewards"].sum().item()) for t in trajectories]

    print(
        f"Saved {len(trajectories)} trajectories to {path} | "
        f"avg_len={np.mean(lengths):.1f} | "
        f"avg_return={np.mean(returns):.1f}"
    )


def collect_all_from_config(config: dict, overwrite: bool):
    env_cfg = config["env"]
    expert_cfg = config["expert"]

    env_name = env_cfg["name"]
    env_id = env_cfg["id"]
    max_steps = int(env_cfg["max_steps"])

    expert_algo = expert_cfg["algo"].lower()
    expert_path = Path(expert_cfg["save_path"])

    if expert_path.suffix != ".zip":
        expert_path = expert_path.with_suffix(".zip")

    if not expert_path.exists():
        raise FileNotFoundError(
            f"Expert checkpoint not found: {expert_path}. "
            f"Train it first with scripts/train_expert.py."
        )

    split_cfg = config.get("trajectory_collection")

    if split_cfg is None:
        raise ValueError("Config must contain `trajectory_collection` section.")

    output_dir = Path(split_cfg.get("output_dir", f"data/{env_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)

    expert_model = load_expert_model(expert_algo, expert_path)
    expert_policy = SB3PolicyWrapper(expert_model, env.action_space)
    random_policy = RandomPolicy(env.action_space)

    print("=" * 80)
    print(f"Collecting trajectories for {env_name}")
    print(f"Env ID: {env_id}")
    print(f"Expert algo: {expert_algo.upper()}")
    print(f"Expert path: {expert_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    saved_files = {}

    # Expert trajectories: train / valid / test
    split_names = ["train", "valid", "test"]
    for split_name in split_names:
        if split_name not in split_cfg:
            continue

        n = int(split_cfg[split_name]["n"])
        seed_start = int(split_cfg[split_name]["seed"])

        out_path = output_dir / f"expert_{split_name}_trajs.pt"

        if out_path.exists() and not overwrite:
            print(f"Skip existing file: {out_path}")
        else:
            expert_trajs = collect_trajectories(
                env=env,
                policy=expert_policy,
                n_trajectories=n,
                max_steps=max_steps,
                seed_start=seed_start,
                desc=f"expert {split_name}",
            )

            save_trajectories(out_path, expert_trajs)

        saved_files[f"expert_{split_name}"] = str(out_path)

    # Random trajectories
    split_names = ["valid", "test"]
    random_trajs_seed = int(split_cfg.get("random_trajs_seed", 1_000_000))
    for split_name in split_names:
        if split_name not in split_cfg:
            continue

        n = int(split_cfg[split_name]["n"])
        seed_start = int(split_cfg[split_name]["seed"]) + random_trajs_seed

        out_path = output_dir / f"random_{split_name}_trajs.pt"

        if out_path.exists() and not overwrite:
            print(f"Skip existing file: {out_path}")
        else:
            random_trajs = collect_trajectories(
                env=env,
                policy=random_policy,
                n_trajectories=n,
                max_steps=max_steps,
                seed_start=seed_start,
                desc=f"random {split_name}",
            )

            save_trajectories(out_path, random_trajs)

        saved_files[f"random_{split_name}"] = str(out_path)

    metadata = {
        "env_name": env_name,
        "env_id": env_id,
        "max_steps": max_steps,
        "expert_algo": expert_algo,
        "expert_path": str(expert_path),
        "output_dir": str(output_dir),
        "splits": {
            split_name: split_cfg[split_name]
            for split_name in split_names
            if split_name in split_cfg
        },
        "files": saved_files,
        "trajectory_format": {
            "states": "[T, state_dim], torch.float32",
            "actions": "[T] for discrete or [T, action_dim] for continuous",
            "env_rewards": "[T], torch.float32",
            "dones": "[T], torch.float32",
            "seed": "int",
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


def main():
    args = parse()

    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)

    collect_all_from_config(config, args.overwrite)


if __name__ == "__main__":
    main()
