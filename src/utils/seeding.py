import os
import random

import numpy as np
import torch
from gymnasium import Env


def set_random_seed(seed: int, deterministic_torch: bool = False) -> None:
    """
    Set global random seeds for Python, NumPy and PyTorch.

    Args:
        seed: Random seed.
        deterministic_torch: If True, enables deterministic PyTorch behavior
            where possible. This can make training slower.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def set_env_seed(env: Env, seed: int) -> None:
    env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
