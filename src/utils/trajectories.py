import numpy as np
import torch
from tqdm import tqdm

from src.utils.policies import Policy
from src.utils.env import Environment


def collect_trajectories(
    env: Environment,
    policy: Policy,
    n: int,
    max_steps: int = 1000,
    deterministic: bool = False,
    desc: str = "collect trajs",
    verbose: bool = True,
):
    if not hasattr(policy, "sample") or not callable(policy.sample):
        raise TypeError("policy must have a callable method `sample(state)`")

    trajs = []

    for _ in tqdm(range(n), desc=desc, leave=False, disable=not verbose):
        states = []
        actions = []
        rewards = []

        state = env.reset()

        for _ in range(max_steps):
            with torch.no_grad():
                action = policy.sample(states=state, deterministic=deterministic)

            next_state, reward, done = env.step(action)

            states.append(state.detach())
            actions.append(action.detach())
            rewards.append(reward.detach())

            state = next_state

            if done.item():
                break

        trajs.append(
            {
                "states": torch.stack(states),
                "actions": torch.stack(actions),
                "rewards": torch.stack(rewards),
            }
        )

    return trajs


def trajectory_return(traj: dict) -> float:
    rewards = traj["rewards"]

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


def discount_weights(T: int, gamma: float, device: str | torch.device = "cpu", dtype=torch.float32) -> torch.Tensor:
    ts = torch.arange(T, dtype=dtype, device=device)
    return torch.pow(torch.tensor(gamma, dtype=dtype, device=device), ts)
