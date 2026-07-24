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
def policy_entropy(policy, trajs) -> float:
    """Estimate mean squashed-action entropy from sampled policy trajectories."""
    device = next(policy.parameters()).device

    negative_log_prob_sum = 0.0
    n_steps = 0
    for traj in trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)
        log_probs = policy.log_prob(states, actions)

        negative_log_prob_sum -= log_probs.sum().item()
        n_steps += log_probs.numel()

    if n_steps == 0:
        return float("nan")

    return negative_log_prob_sum / n_steps


@torch.no_grad()
def outer_loss(
    policy,
    expert_trajs,
    discount: float,
    normalize_discounted_sums: bool = False,
) -> float:
    if not (0.0 < discount <= 1.0):
        raise ValueError(f"`discount` must satisfy 0 < discount <= 1, got {discount}.")
    
    device = next(policy.parameters()).device

    loss = 0.0
    for traj in expert_trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)

        log_probs = policy.log_prob(states, actions)
        weights = discount_weights(log_probs.size(0), discount, device)
        if normalize_discounted_sums:
            weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        loss += (weights * log_probs).sum().item()

    return -loss / len(expert_trajs)


@torch.no_grad()
def observation_outer_loss(policy, expert_trajs) -> float:
    """Return the mean expert negative log likelihood per observation."""
    if not expert_trajs:
        raise ValueError("At least one expert trajectory is required.")

    device = next(policy.parameters()).device
    negative_log_prob_sum = 0.0
    observation_count = 0

    for traj in expert_trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)
        log_probs = policy.log_prob(states, actions)

        negative_log_prob_sum -= log_probs.sum().item()
        observation_count += log_probs.numel()

    if observation_count == 0:
        raise ValueError("Expert trajectories must contain at least one observation.")

    return negative_log_prob_sum / observation_count


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
    step_rewards = []
    lengths = []

    for traj in trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)
        T = states.size(0)

        rewards = reward(states, actions)  # (T,)
        weights = discount_weights(T, discount, device).to(dtype=rewards.dtype)
        traj_return = (weights * rewards).sum()

        returns.append(traj_return.item())
        step_rewards.extend(rewards.detach().cpu().numpy().tolist())
        lengths.append(float(T))

    returns = np.asarray(returns, dtype=np.float64)
    step_rewards = np.asarray(step_rewards, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)

    def summary(name: str, values: np.ndarray) -> dict[str, float]:
        quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        return {
            f"{name}_mean": float(values.mean()),
            f"{name}_std": float(values.std()),
            f"{name}_q05": float(quantiles[0]),
            f"{name}_q25": float(quantiles[1]),
            f"{name}_q50": float(quantiles[2]),
            f"{name}_q75": float(quantiles[3]),
            f"{name}_q95": float(quantiles[4]),
        }

    return {
        **summary("return", returns),
        **summary("step", step_rewards),
        "len_mean": float(lengths.mean()),
        "len_std": float(lengths.std()),
    }


@torch.no_grad()
def learned_reward_observation_stats(reward, trajs) -> dict[str, float]:
    """Summarize learned rewards over the pooled observations without discounting."""
    if not trajs:
        raise ValueError("At least one trajectory is required.")

    device = next(reward.parameters()).device
    observation_rewards = []

    for traj in trajs:
        states = to_device(traj["states"], device)
        actions = to_device(traj["actions"], device)
        observation_rewards.append(reward(states, actions).detach().cpu())

    values = torch.cat(observation_rewards).numpy().astype(np.float64, copy=False)
    if values.size == 0:
        raise ValueError("Trajectories must contain at least one observation.")

    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "observation_mean": float(values.mean()),
        "observation_std": float(values.std()),
        "observation_q05": float(quantiles[0]),
        "observation_q25": float(quantiles[1]),
        "observation_q50": float(quantiles[2]),
        "observation_q75": float(quantiles[3]),
        "observation_q95": float(quantiles[4]),
        "observation_count": float(values.size),
    }
