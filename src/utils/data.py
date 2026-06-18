from pathlib import Path
from typing import Any

import torch

Trajectory = dict[str, Any]
Trajectories = list[Trajectory]


def load_trajectories(
    path: str | Path,
    map_location: str | torch.Device = "cpu",
) -> Trajectories:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    return torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )
