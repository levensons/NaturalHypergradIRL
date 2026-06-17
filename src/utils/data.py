from pathlib import Path
from typing import Any, Tuple, Dict, List
import torch

Trajectory = Dict[str, Any]
Trajectories = List[Trajectory]


def load_trajectories(path: str | Path, map_location: str | torch.device = "cpu") -> Trajectories:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    return torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )


def load_expert_train_trajectories(config: dict, map_location: str | torch.device = "cpu") -> Tuple[Trajectories, Path]:
    env_name = config["env"]["name"]
    data_cfg = config["data"]

    path = Path(
        data_cfg.get(
            "expert_train_trajs",
            f"data/{env_name}/expert_train_trajs.pt",
        )
    )

    return load_trajectories(path, map_location=map_location), path


def load_expert_validation_trajectories(
    config: dict, map_location: str | torch.device = "cpu"
) -> Tuple[Trajectories, Path]:
    env_name = config["env"]["name"]
    data_cfg = config["data"]

    path = Path(
        data_cfg.get(
            "expert_valid_trajs",
            f"data/{env_name}/expert_valid_trajs.pt",
        )
    )

    return load_trajectories(path, map_location=map_location), path


def load_random_validation_trajectories(
    config: dict, map_location: str | torch.device = "cpu"
) -> Tuple[Trajectories, Path]:
    env_name = config["env"]["name"]
    data_cfg = config["data"]

    path = Path(
        data_cfg.get(
            "random_valid_trajs",
            f"data/{env_name}/random_valid_trajs.pt",
        )
    )

    return load_trajectories(path, map_location=map_location), path


def load_validation_trajectories(
    config: dict, map_location: str | torch.device = "cpu"
) -> Tuple[Trajectories, Trajectories]:
    expert_valid_trajs, _ = load_expert_validation_trajectories(
        config,
        map_location=map_location,
    )
    random_valid_trajs, _ = load_random_validation_trajectories(
        config,
        map_location=map_location,
    )

    return expert_valid_trajs, random_valid_trajs


def load_expert_test_trajectories(config: dict, map_location: str | torch.device = "cpu") -> Tuple[Trajectories, Path]:
    env_name = config["env"]["name"]
    data_cfg = config["data"]

    path = Path(
        data_cfg.get(
            "expert_test_trajs",
            f"data/{env_name}/expert_test_trajs.pt",
        )
    )

    return load_trajectories(path, map_location=map_location), path


def load_random_test_trajectories(config: dict, map_location: str | torch.device = "cpu") -> Tuple[Trajectories, Path]:
    env_name = config["env"]["name"]
    data_cfg = config["data"]

    path = Path(
        data_cfg.get(
            "random_test_trajs",
            f"data/{env_name}/random_test_trajs.pt",
        )
    )

    return load_trajectories(path, map_location=map_location), path
