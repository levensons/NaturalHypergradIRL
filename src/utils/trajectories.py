from typing import Any
import numpy as np
import torch
from gymnasium import Env, spaces
from tqdm import tqdm

from src.utils.policies import Policy


def _check_policy(policy: Policy) -> None:
    if not hasattr(policy, "sample_action") or not callable(policy.sample_action):
        raise TypeError("policy must have a callable method `sample_action(state)`")


def _prepare_action(env: Env, action: Any):
    if isinstance(env.action_space, spaces.Box):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        action = action.astype(np.float32, copy=False)
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        return action, action_tensor

    if isinstance(env.action_space, spaces.Discrete):
        action = int(action)
        action_tensor = torch.tensor(action, dtype=torch.long)
        return action, action_tensor

    raise TypeError(f"Unsupported action space: {type(env.action_space)}")


def collect_trajectories(
    env: Env,
    policy: Policy,
    n: int,
    max_steps: int = 1000,
    desc: str = "collect trajs",
    verbose: bool = True,
):
    _check_policy(policy)

    trajs = []

    for _ in tqdm(range(n), desc=desc, leave=False, disable=not verbose):
        states = []
        actions = []
        env_rewards = []

        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy.sample_action(state)
            action, action_tensor = _prepare_action(env, action)

            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(torch.as_tensor(state, dtype=torch.float32))
            actions.append(action_tensor)
            env_rewards.append(float(reward))

            state = next_state

            if terminated or truncated:
                break

        trajs.append(
            {
                "states": torch.stack(states),
                "actions": torch.stack(actions),
                "env_rewards": torch.tensor(env_rewards, dtype=torch.float32),
            }
        )

    return trajs


def trajectory_return(traj: dict) -> float:
    rewards = traj["env_rewards"]

    if isinstance(rewards, torch.Tensor):
        return float(rewards.sum().item())

    return float(sum(rewards))


def mean_trajectory_length(trajs) -> float:
    return float(np.mean([len(t["states"]) for t in trajs]))


def mean_trajectory_return(trajs) -> float:
    return float(np.mean([trajectory_return(t) for t in trajs]))


def trajectory_summary(trajs) -> dict:
    return {
        "len": mean_trajectory_length(trajs),
        "return": mean_trajectory_return(trajs),
    }
