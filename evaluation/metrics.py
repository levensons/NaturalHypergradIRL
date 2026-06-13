import numpy as np
import torch
import torch.nn as nn


class RankCorr(nn.Module):
    """Ранговая корреляция Пирсона между истинными и предсказанными возвратами."""
    @staticmethod
    def rankdata(x: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x.float()
        y = y.float()
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt(torch.sum(x**2) * torch.sum(y**2))
        if denom < eps:
            return torch.tensor(torch.nan)
        return torch.sum(x * y) / denom

    @torch.no_grad()
    def forward(self, reward, trajs) -> float:
        reward.eval()
        env_returns, learned_returns = [], []
        for traj in trajs:
            env_return = torch.tensor(traj["env_rewards"], dtype=torch.float32).sum()
            env_returns.append(env_return)
            learned_return = reward.trajectory_return(traj["states"], traj["actions"])
            learned_returns.append(learned_return)
        env_returns = torch.stack(env_returns)
        learned_returns = torch.stack(learned_returns)
        env_ranks = self.rankdata(env_returns)
        learned_ranks = self.rankdata(learned_returns)
        return self.pearson_corr(env_ranks, learned_ranks).item()


class PolicyNLL(nn.Module):
    """Средний отрицательный log-likelihood политики на экспертных траекториях."""
    @torch.no_grad()
    def forward(self, policy, expert_trajs) -> float:
        policy.eval()
        nll = 0.0
        for traj in expert_trajs:
            log_probs = policy.log_prob(traj["states"], traj["actions"])
            nll += -log_probs.sum().item()
        return nll / len(expert_trajs)


class OuterLoss(nn.Module):
    """Внешний лосс: средний -Σ log π на экспертных траекториях."""

    def forward(self, policy, expert_trajs) -> torch.Tensor:
        losses = []
        for traj in expert_trajs:
            log_probs = policy.log_prob(traj["states"], traj["actions"])
            losses.append(-log_probs.sum())
        return torch.stack(losses).mean()


class InnerLoss(nn.Module):
    """Внутренний лосс: средний Σ(log π - R_φ) на агентских траекториях."""

    def forward(self, policy, reward, trajs) -> torch.Tensor:
        losses = []
        for traj in trajs:
            states = traj["states"]
            actions = traj["actions"]
            log_probs = policy.log_prob(states, actions)
            discounted_rewards = reward.discounted_rewards(states, actions)
            losses.append((log_probs - discounted_rewards).sum())
        return torch.stack(losses).mean()


def env_reward(trajs) -> float:
    """Средний недисконтированный возврат по списку траекторий."""
    return float(np.mean([sum(t["env_rewards"]) for t in trajs]))
