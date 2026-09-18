from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import torch
from tqdm import tqdm

from src.utils.policies import Policy
from src.utils.env import Environment


def collect_trajectory(
    env: Environment,
    policy: Policy,
    deterministic: bool = False,
) -> dict:
    states = []
    actions = []
    rewards = []
    next_states = []
    terminateds = []

    state = env.reset()

    while True:
        with torch.no_grad():
            action = policy.sample(states=state, deterministic=deterministic)

        next_state, reward, terminated, truncated = env.step(action)

        states.append(state.detach())
        actions.append(action.detach())
        rewards.append(reward.detach())
        next_states.append(next_state.detach())
        terminateds.append(terminated.detach())

        state = next_state

        if terminated.item() or truncated.item():
            break

    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "rewards": torch.stack(rewards),
        "next_states": torch.stack(next_states),
        "terminateds": torch.stack(terminateds),
    }


def collect_trajectories(
    env: Environment,
    policy: Policy,
    n: int,
    deterministic: bool = False,
    desc: str = "collect trajs",
    verbose: bool = True,
):
    if not hasattr(policy, "sample") or not callable(policy.sample):
        raise TypeError("policy must have a callable method `sample(state)`")

    trajs = []

    for _ in tqdm(range(n), desc=desc, leave=False, disable=not verbose):
        trajs.append(collect_trajectory(env, policy, deterministic))

    return trajs


def collect_trajectories_parallel(
    envs,
    policy,
    n: int,
    deterministic: bool = False,
    desc: str = "collect trajs",
    verbose: bool = True,
):
    if not envs:
        raise ValueError("`envs` must contain at least one environment.")

    n_workers = len(envs)

    base = n // n_workers
    remainder = n % n_workers
    counts = [base + (worker_id < remainder) for worker_id in range(n_workers)]

    progress_lock = Lock()

    with tqdm(total=n, desc=desc, leave=False, disable=not verbose) as progress:

        def collect(env, count: int):
            trajectories = []

            for _ in range(count):
                traj = collect_trajectory(env, policy, deterministic)
                trajectories.append(traj)

                with progress_lock:
                    progress.update(1)

            return trajectories

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(collect, env, count)
                for env, count in zip(envs, counts) if count > 0
            ]

            trajectories = []
            for future in futures:
                trajectories.extend(future.result())

    return trajectories


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


def discount_weights(
    trajectory_lengths: int | Sequence[int],
    gamma: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32
):
    if isinstance(trajectory_lengths, int):
        gamma = torch.tensor(gamma, dtype=dtype, device=device)
        weights = torch.pow(gamma, torch.arange(trajectory_lengths, device=device))
        return weights

    gamma = torch.tensor(gamma, dtype=dtype, device=device)
    weights = [torch.pow(gamma, torch.arange(ts, device=device)) for ts in trajectory_lengths]
    return weights
