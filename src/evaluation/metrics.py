import numpy as np
import torch

from src.utils.torch import to_device
from src.utils.trajectories import discount_weights


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    sorted_x, order = torch.sort(x)
    ranks = torch.empty_like(x, dtype=torch.float32)
    n = x.numel()
    i = 0

    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1

        avg_rank = 0.5 * (i + j)
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.pow(x, 2).sum() * torch.pow(y, 2).sum())

    if denom < eps:
        return torch.tensor(torch.nan)

    return torch.sum(x * y) / denom


@torch.no_grad()
def rank_corr(reward, trajs) -> float:
    device = next(reward.parameters()).device
    reward.eval()

    env_returns, learned_returns = [], []

    for traj in trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)
        env_rewards = to_device(traj["env_rewards"], device)

        env_return = env_rewards.sum()
        learned_return = reward.trajectory_return(states, actions)

        env_returns.append(env_return)
        learned_returns.append(learned_return)

    env_returns = torch.stack(env_returns)
    learned_returns = torch.stack(learned_returns)

    env_ranks = _rankdata(env_returns)
    learned_ranks = _rankdata(learned_returns)

    return _pearson_corr(env_ranks, learned_ranks).item()


@torch.no_grad()
def policy_nll(policy, expert_trajs) -> float:
    device = next(policy.parameters()).device
    policy.eval()

    nll = 0.0
    for traj in expert_trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)

        log_probs = policy.log_prob(states, actions)
        nll += log_probs.sum().item()

    return -nll / len(expert_trajs)


@torch.no_grad()
def outer_loss(policy, expert_trajs, discount: float) -> float:
    if not (0.0 < discount <= 1.0):
        raise ValueError(f"`discount` must satisfy 0 < discount <= 1, got {discount}.")
    
    device = next(policy.parameters()).device

    loss = 0.0
    for traj in expert_trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)

        log_probs = policy.log_prob(states, actions)
        weights = discount_weights(log_probs.size(0), discount, device)
        loss += (weights * log_probs).sum().item()

    return -loss / len(expert_trajs)


@torch.no_grad()
def inner_loss(policy, reward, trajs, discount: float, alpha: float = 1.0) -> float:
    if not (0.0 < discount <= 1.0):
        raise ValueError(f"`discount` must satisfy 0 < discount <= 1, got {discount}.")
    
    if not (0.0 < alpha):
        raise ValueError(f"`alpha` must satisfy alpha > 0, got {alpha}.")
    
    device = next(policy.parameters()).device

    loss = 0.0
    for traj in trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)

        log_probs = policy.log_prob(states, actions)
        rewards = reward.rewards(states, actions)
        weights = discount_weights(log_probs.size(0), discount, device)
        loss += (weights * (alpha * log_probs - rewards)).sum().item()

    return loss / len(trajs)


@torch.no_grad()
def env_reward(trajs) -> float:
    total = 0.0
    for traj in trajs:
        total += torch.as_tensor(traj["env_rewards"]).sum().item()
    return total / len(trajs)


@torch.no_grad()
def learned_reward_stats(reward, trajs, discount: float = 1.0) -> dict[str, float]:
    device = next(reward.parameters()).device

    returns = []
    means = []
    lengths = []

    for traj in trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)
        T = states.size(0)

        rewards = reward(states, actions)  # (T,)
        weights = discount_weights(T, discount, device).to(dtype=rewards.dtype)
        traj_return = (weights * rewards).sum()

        returns.append(traj_return.item())
        means.append(rewards.mean().item())
        lengths.append(float(T))

    returns = np.asarray(returns, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)

    return {
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "step_mean": float(means.mean()),
        "len_mean": float(lengths.mean()),
    }
