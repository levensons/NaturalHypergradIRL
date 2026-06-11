import argparse
from pathlib import Path

import gymnasium as gym
import yaml
from stable_baselines3 import PPO, SAC


def load_config(path: str) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_env(env, seed: int):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def get_model(algo: str, env, params: dict, verbose: int):
    params = dict(params)

    policy = params.pop("policy", "MlpPolicy")

    if algo == "sac":
        return SAC(policy, env, **params, verbose=verbose)

    if algo == "ppo":
        return PPO(policy, env, **params, verbose=verbose)

    raise ValueError(f"Unknown expert algorithm: {algo}")


def train_expert_from_config(config: dict, verbose: int, overwrite: bool):
    env_cfg = config["env"]
    expert_cfg = config["expert"]

    env_id = env_cfg["id"]
    env_seed = int(env_cfg.get("env_seed", 42))

    algo = expert_cfg["algo"].lower()
    save_path = Path(expert_cfg["save_path"])
    total_timesteps = int(expert_cfg["total_timesteps"])
    expert_params = expert_cfg.get("params", {})

    if save_path.suffix == ".zip":
        save_path = save_path.with_suffix("")

    zip_path = save_path.with_suffix(".zip")

    if zip_path.exists() and not overwrite:
        print(f"Expert already exists: {zip_path}")
        print("Use --overwrite to retrain it.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    seed_env(env, env_seed)

    model = get_model(algo, env, expert_params, verbose)
    model.learn(total_timesteps=total_timesteps)
    model.save(str(save_path))

    env.close()

    print(f"Saved expert to {zip_path}")


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

    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def resolve_config_path(env_name: str | None, config_path: str | None) -> str:
    if config_path is not None:
        return config_path

    if env_name is None:
        raise ValueError("Specify either --env or --config.")

    return f"config/{env_name}.yaml"


def main():
    args = parse()

    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)

    train_expert_from_config(config, args.verbose, args.overwrite)


if __name__ == "__main__":
    main()
