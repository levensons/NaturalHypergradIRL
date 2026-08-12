"""
Train an expert policy for the selected environment.

Usage:
    python -m src.prepare_data.train_expert --env hopper
    python -m src.prepare_data.train_expert --config configs/hopper.yaml

To retrain an existing expert:
    python -m src.prepare_data.train_expert --env hopper --overwrite

The trained expert is saved to config["expert"]["save_path"].
"""

import argparse
from pathlib import Path

from src.utils.config import load_config, resolve_config_path
from src.utils.seeding import set_random_seed
from src.utils.sb3 import init_sb3_model, normalize_sb3_load_path, normalize_sb3_save_path
from src.utils.env import Environment
from envs.lqr.env import LQR, GymLQR

def create_expert_env(env_cfg: dict, seed: int):
    env_name = env_cfg["name"].lower()

    if env_name == "lqr":
        lqr = LQR(
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
            seed=seed,
        )
        return GymLQR(lqr)

    return Environment(env_cfg["id"], seed, int(env_cfg["max_steps"]))

def train_expert_from_config(config: dict, verbose: int, overwrite: bool):
    env_cfg = config["env"]
    expert_cfg = config["expert"]

    env_id = env_cfg["id"]
    env_seed = int(expert_cfg["env_seed"])
    random_seed = int(expert_cfg["random_seed"])

    algo = expert_cfg["algo"].lower()
    save_path = Path(expert_cfg["save_path"])
    total_timesteps = int(expert_cfg["total_timesteps"])
    expert_params = expert_cfg["params"]

    if save_path.suffix == ".zip":
        save_path = save_path.with_suffix("")

    zip_path = save_path.with_suffix(".zip")

    if zip_path.exists() and not overwrite:
        print(f"Expert already exists: {zip_path}")
        print("Use --overwrite to retrain it.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    set_random_seed(random_seed)
    env = create_expert_env(env_cfg, env_seed)

    save_path = normalize_sb3_save_path(expert_cfg["save_path"])
    zip_path = normalize_sb3_load_path(save_path)

    model = init_sb3_model(algo, env, expert_params, verbose)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=50)
    model.save(str(save_path))

    env.close()

    print(f"Saved expert to {zip_path}")


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        choices=["hopper", "cartpole", "lqr"],
        default=None,
        help="Environment name. Used to load config/<env>.yaml.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit path to config YAML. Overrides --env.",
    )

    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def main():
    args = parse()
    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)
    train_expert_from_config(config, args.verbose, args.overwrite)


if __name__ == "__main__":
    main()
