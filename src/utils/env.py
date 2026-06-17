from gymnasium import Env, spaces
from typing import Tuple


def get_env_dims(env: Env) -> Tuple[int, int]:
    """
    Return state and action dimensions for a Gymnasium environment.

    Supports:
        - Box observation spaces
        - Box action spaces
        - Discrete action spaces
    """
    if not isinstance(env.observation_space, spaces.Box):
        raise TypeError(f"Unsupported observation space: {type(env.observation_space)}")

    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, spaces.Box):
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, spaces.Discrete):
        action_dim = env.action_space.n
    else:
        raise TypeError(f"Unsupported action space: {type(env.action_space)}")

    return state_dim, action_dim
