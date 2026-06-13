"""
TTSA-бейзлайн IRL для Hopper (непрерывная среда).

Запуск:
  python -m irl.hopper.ttsa_hopper --env hopper
  python -m irl.hopper.ttsa_hopper --config config/hopper.yaml
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Normal

from evaluation.metrics import InnerLoss, OuterLoss, PolicyNLL, RankCorr
from src.common.config import load_config, resolve_config_path, set_seed
from src.common.logging_utils import get_logger, save_history


class GaussianPolicyNet(nn.Module):
    def __init__(self, state_dim=11, action_dim=3, hidden=64, action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x):
        mu      = self.action_scale * torch.tanh(self.mu_net(x))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std     = torch.exp(log_std).expand_as(mu)
        return mu, std

    def log_prob(self, states, actions):
        mu, std = self.forward(states)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        return Normal(mu, std).log_prob(actions).sum(dim=-1)

    def sample_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mu, std = self.forward(state_t)
            a = Normal(mu, std).sample()
        return np.clip(a.numpy(), -self.action_scale, self.action_scale)


class RewardNet(nn.Module):
    def __init__(self, state_dim=11, gamma=0.99):
        super().__init__()
        self.net   = nn.Linear(state_dim, 1, bias=True)
        self.gamma = gamma

    def base_reward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)

    def discounted_rewards(self, states: torch.Tensor) -> torch.Tensor:
        ts = torch.arange(states.size(0), dtype=torch.float32)
        return self.base_reward(states) * (self.gamma ** ts)

    def discounted_return(self, states: torch.Tensor) -> torch.Tensor:
        return self.discounted_rewards(states).sum()

    def trajectory_return(self, states: torch.Tensor, actions=None) -> torch.Tensor:
        """Интерфейс для evaluation.metrics.RankCorr (actions игнорируются)."""
        return self.discounted_return(states)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.base_reward(states)


# Алиасы для evaluate.py dispatcher
Policy = GaussianPolicyNet
Reward = RewardNet


class SB3PolicyWrapper:
    def __init__(self, model, action_scale=1.0):
        self.model        = model
        self.action_scale = action_scale

    def sample_action(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def eval(self):
        pass

    def train(self):
        pass


class TTSANHD(nn.Module):
    def __init__(self, policy: GaussianPolicyNet, reward: RewardNet,
                 n_cg_steps: int = 10, reg: float = 1e-3):
        super().__init__()
        self.policy     = policy
        self.reward     = reward
        self.n_cg_steps = n_cg_steps
        self.reg        = reg

    def score(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def reinforce_grad(self, trajs) -> torch.Tensor:
        d    = sum(p.numel() for p in self.policy.parameters())
        ells = []
        for traj in trajs:
            s, a = traj["states"], traj["actions"]
            T    = len(s)
            gammas = torch.tensor([self.reward.gamma ** t for t in range(T)])
            with torch.no_grad():
                ell = (gammas * (self.policy.log_prob(s, a) - self.reward(s))).sum().item()
            ells.append(ell)
        baseline = float(np.mean(ells))
        grad = torch.zeros(d)
        for traj, ell in zip(trajs, ells):
            grad += (ell - baseline) * self.score(traj["states"], traj["actions"])
        return grad / len(trajs)

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        d    = sum(p.numel() for p in self.policy.parameters())
        grad = torch.zeros(d)
        for traj in expert_trajs:
            grad += self.score(traj["states"], traj["actions"])
        return -grad / len(expert_trajs)

    def cross_derivative(self, trajs) -> torch.Tensor:
        d_theta = sum(p.numel() for p in self.policy.parameters())
        d_phi   = sum(p.numel() for p in self.reward.parameters())
        cross   = torch.zeros(d_theta, d_phi)
        for traj in trajs:
            s, a       = traj["states"], traj["actions"]
            s_theta    = self.score(s, a)
            reward_sum = self.reward.discounted_return(s)
            grads_phi  = torch.autograd.grad(reward_sum, self.reward.parameters())
            g_phi = flat_grad(grads_phi)
            cross += torch.outer(s_theta, g_phi)
        return -cross / len(trajs)

    def _fisher_vector_product(self, trajs, u: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(u)
        for traj in trajs:
            s_theta = self.score(traj["states"], traj["actions"])
            result += s_theta * (s_theta @ u)
        result /= len(trajs)
        return result + self.reg * u

    def _conjugate_gradient_solve(self, trajs, g: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
        v = torch.zeros_like(g)
        r = g.clone()
        p = r.clone()
        r_dot_r = (r * r).sum()
        if r_dot_r < tol:
            return v
        for _ in range(self.n_cg_steps):
            Fp  = self._fisher_vector_product(trajs, p)
            pFp = (p * Fp).sum()
            if pFp <= 0:
                break
            alpha   = r_dot_r / pFp
            v       = v + alpha * p
            r       = r - alpha * Fp
            new_r   = (r * r).sum()
            if new_r < tol:
                break
            p       = r + (new_r / r_dot_r) * p
            r_dot_r = new_r
        return v

    def forward(self, expert_trajs, agent_trajs) -> torch.Tensor:
        g = self.outer_grad(expert_trajs)
        v = self._conjugate_gradient_solve(agent_trajs, g)
        H = self.cross_derivative(agent_trajs)
        return -H.T @ v


def flat_grad(grads):
    return torch.cat([g.flatten() for g in grads])


def assign_flat_gradients(module: nn.Module, flat_grad_vec: torch.Tensor):
    i = 0
    for p in module.parameters():
        n = p.numel()
        grad_chunk = flat_grad_vec[i : i + n]
        if grad_chunk.numel() != n:
            raise ValueError("Flat gradient has incorrect size: not enough elements.")
        p.grad = grad_chunk.reshape(p.shape).clone()
        i += n
    if i != flat_grad_vec.numel():
        raise ValueError("Flat gradient has incorrect size: too many elements.")


def _safe_clip_grad(grad: torch.Tensor, max_norm: float):
    if not torch.isfinite(grad).all():
        return torch.zeros_like(grad), False
    norm = grad.norm()
    if norm.item() == 0.0:
        return grad, True
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad, True


def collect_trajectories(env, policy: GaussianPolicyNet, n: int, max_steps: int = 1000):
    trajs = []
    for _ in range(n):
        states, actions, env_rewards = [], [], []
        s, _ = env.reset()
        for _ in range(max_steps):
            a = policy.sample_action(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            states.append(torch.tensor(s, dtype=torch.float32))
            actions.append(torch.tensor(a, dtype=torch.float32))
            env_rewards.append(float(r))
            s = s_next
            if terminated or truncated:
                break
        trajs.append({"states": torch.stack(states), "actions": torch.stack(actions),
                      "env_rewards": env_rewards})
    return trajs


def train_ttsa(env, expert_trajs, config: dict, logger=None) -> dict:
    """TTSA bilevel оптимизация для Hopper. Алгоритм не изменён."""
    train_cfg = config["training"]
    log_cfg   = config.get("logging", {})

    n_iterations    = int(train_cfg.get("n_outer_steps", 2000))
    n_traj_per_step = int(train_cfg.get("n_agent_traj", 70))
    alpha_inner     = float(train_cfg.get("lr_inner", 3e-3))
    beta_outer      = float(train_cfg.get("lr_outer", 3e-4))
    gamma           = float(config.get("reward", {}).get("gamma", 0.99))
    metrics_every   = int(log_cfg.get("log_every", 50))

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy     = GaussianPolicyNet(state_dim=state_dim, action_dim=action_dim)
    reward_net = RewardNet(state_dim=state_dim, gamma=gamma)
    ttsa       = TTSANHD(policy, reward_net, n_cg_steps=20, reg=0.1)

    inner_optimizer = torch.optim.SGD(policy.parameters(),     lr=alpha_inner)
    outer_optimizer = torch.optim.SGD(reward_net.parameters(), lr=beta_outer)

    outer_loss_fn = OuterLoss()
    inner_loss_fn = InnerLoss()

    history = {
        "l_outer": [], "l_inner": [], "env_reward": [],
        "hypgrad_norm": [], "agent_len": [], "iter_time": [],
    }

    t_start = time.time()

    for k in range(1, n_iterations + 1):
        t_iter = time.time()
        agent_trajs = collect_trajectories(env, policy, n=n_traj_per_step)

        inner_grad, ok_in = _safe_clip_grad(ttsa.reinforce_grad(agent_trajs), max_norm=5.0)
        if not ok_in:
            msg = f"[iter {k}] WARNING: NaN/Inf in inner_grad, skipping inner step"
            logger.warning(msg) if logger else print(msg)
        else:
            inner_optimizer.zero_grad()
            assign_flat_gradients(policy, inner_grad)
            inner_optimizer.step()

        hypgrad = ttsa(expert_trajs, agent_trajs)
        hg_norm_before_clip = (
            hypgrad.norm().item() if torch.isfinite(hypgrad).all() else float("inf")
        )
        hypgrad, ok_out = _safe_clip_grad(hypgrad, max_norm=1.0)
        if not ok_out:
            msg = f"[iter {k}] WARNING: NaN/Inf in hypgrad, skipping outer step"
            logger.warning(msg) if logger else print(msg)
        else:
            outer_optimizer.zero_grad()
            assign_flat_gradients(reward_net, hypgrad)
            outer_optimizer.step()

        if k % metrics_every == 0 or k == 1:
            l_out        = outer_loss_fn(policy, expert_trajs).item()
            l_in         = inner_loss_fn(policy, reward_net, agent_trajs).item()
            agent_len    = float(np.mean([len(t["states"]) for t in agent_trajs]))
            env_r        = float(np.mean([sum(t["env_rewards"]) for t in agent_trajs]))
            iter_elapsed = time.time() - t_iter

            history["l_outer"].append(l_out)
            history["l_inner"].append(l_in)
            history["hypgrad_norm"].append(hg_norm_before_clip)
            history["agent_len"].append(agent_len)
            history["env_reward"].append((k, env_r))
            history["iter_time"].append(iter_elapsed)

            row = (f"{k:>5} | {l_out:>10.3f} | {l_in:>10.3f} | "
                   f"{env_r:>8.1f} | {agent_len:>5.1f} | "
                   f"{hg_norm_before_clip:>8.4f} | {iter_elapsed:>6.1f}s")
            logger.info(row) if logger else print(row)

    t_total = time.time() - t_start
    msg = f"Total time: {t_total:.1f}s ({t_total/60:.1f} min)"
    logger.info(msg) if logger else print(msg)

    return history


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTSA IRL — Hopper")
    p.add_argument("--env", choices=["hopper"], default="hopper")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse()
    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)

    train_cfg = config["training"]
    log_cfg   = config.get("logging", {})
    data_cfg  = config.get("data", {})

    set_seed(int(train_cfg.get("seed", 42)))

    log_dir = log_cfg.get("log_dir", "logs")
    logger  = get_logger("ttsa_hopper", log_dir=log_dir)

    env = gym.make(config["env"]["id"])
    env.reset(seed=int(train_cfg.get("seed", 42)))

    expert_train_path = data_cfg.get("expert_train_trajs", "data/hopper/expert_train_trajs.pt")
    all_expert_trajs = torch.load(expert_train_path, map_location="cpu", weights_only=False)
    n_expert_traj = int(train_cfg.get("n_expert_traj", 1000))
    if len(all_expert_trajs) < n_expert_traj:
        logger.warning(
            f"Запрошено {n_expert_traj} экспертных траекторий, "
            f"доступно {len(all_expert_trajs)} → используем все."
        )
        n_expert_traj = len(all_expert_trajs)
    expert_trajs = all_expert_trajs[:n_expert_traj]
    logger.info(f"Загружено {len(expert_trajs)} экспертных траекторий из {expert_train_path}")

    logger.info("=== TTSA Hopper: начинаю оптимизацию ===")
    history = train_ttsa(env, expert_trajs, config, logger=logger)

    report_path = (
        Path(log_cfg.get("report_dir", "reports")) / "ttsa_hopper_history.json"
    )
    save_history(history, str(report_path))
    logger.info(f"История сохранена в {report_path}")

    env.close()


if __name__ == "__main__":
    main()
