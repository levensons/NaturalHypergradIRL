from pathlib import Path
from stable_baselines3 import PPO, SAC


def normalize_sb3_load_path(path: str | Path) -> Path:
    path = Path(path)

    if path.suffix != ".zip":
        path = path.with_suffix(".zip")

    return path


def normalize_sb3_save_path(path: str | Path) -> Path:
    path = Path(path)

    if path.suffix == ".zip":
        path = path.with_suffix("")

    return path


def init_sb3_model(algo: str, env, params: dict, verbose: int = 1):
    algo = algo.lower()
    params = dict(params)

    policy = params.pop("policy", "MlpPolicy")

    if algo == "sac":
        return SAC(policy, env, **params, verbose=verbose)

    if algo == "ppo":
        return PPO(policy, env, **params, verbose=verbose)

    raise ValueError(f"Unknown SB3 algorithm: {algo}")


def load_sb3_model(algo: str, path: str | Path):
    algo = algo.lower()
    path = normalize_sb3_load_path(path)

    if not path.exists():
        raise FileNotFoundError(f"SB3 checkpoint not found: {path}")

    if algo == "sac":
        return SAC.load(str(path))

    if algo == "ppo":
        return PPO.load(str(path))

    raise ValueError(f"Unknown SB3 algorithm: {algo}")
