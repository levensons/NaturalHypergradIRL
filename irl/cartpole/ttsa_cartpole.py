"""
TTSA-бейзлайн IRL для CartPole (дискретная среда).

Запуск:
  python -m irl.cartpole.ttsa_cartpole --env cartpole
  python -m irl.cartpole.ttsa_cartpole --config config/cartpole.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical

from evaluation.metrics import inner_loss, outer_loss
from src.common.config import load_config, resolve_config_path, set_seed
from src.common.logging_utils import get_logger, save_history


class PolicyNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

    def log_prob(self, states, actions):
        return Categorical(logits=self.forward(states)).log_prob(actions)

    def sample_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        return Categorical(logits=self.forward(state_t)).sample().item()


class RewardNet(nn.Module):
    def __init__(self, state_dim=4, gamma=0.99):
        super().__init__()
        self.net = nn.Linear(state_dim, 1, bias=True)
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


Policy = PolicyNet
Reward = RewardNet


class TTSANHD(nn.Module):
    def __init__(self, policy: PolicyNet, reward: RewardNet,
                 n_cg_steps: int = 10, reg: float = 1e-3):
        super().__init__()
        self.policy = policy
        self.reward = reward
        self.n_cg_steps = n_cg_steps
        self.reg = reg

    def score(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def reinforce_grad(self, trajs) -> torch.Tensor:
        d = sum(p.numel() for p in self.policy.parameters())
        ells = []
        for traj in trajs:
            s, a = traj["states"], traj["actions"]
            T = len(s)
            gammas = torch.tensor([self.reward.gamma ** t for t in range(T)])
            with torch.no_grad():
                ell = (gammas * (self.policy.log_prob(s, a) - self.reward(s))).sum().item()
            ells.append(ell)
        baseline = float(np.mean(ells))
        grad = torch.zeros(d)
        for traj, ell in zip(trajs, ells):
            s, a = traj["states"], traj["actions"]
            grad += (ell - baseline) * self.score(s, a)
        return grad / len(trajs)

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        d = sum(p.numel() for p in self.policy.parameters())
        grad = torch.zeros(d)
        for traj in expert_trajs:
            grad += self.score(traj["states"], traj["actions"])
        return -grad / len(expert_trajs)

    def cross_derivative(self, trajs) -> torch.Tensor:
        d_theta = sum(p.numel() for p in self.policy.parameters())
        d_phi   = sum(p.numel() for p in self.reward.parameters())
        cross   = torch.zeros(d_theta, d_phi)
        for traj in trajs:
            s, a = traj["states"], traj["actions"]
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
            alpha = r_dot_r / pFp
            v = v + alpha * p
            r = r - alpha * Fp
            new_r_dot_r = (r * r).sum()
            if new_r_dot_r < tol:
                break
            beta    = new_r_dot_r / r_dot_r
            p       = r + beta * p
            r_dot_r = new_r_dot_r
        return v

    def forward(self, expert_trajs, agent_trajs) -> torch.Tensor:
        g = self.outer_grad(expert_trajs)
        v = self._conjugate_gradient_solve(agent_trajs, g)
        H = self.cross_derivative(agent_trajs)
        return -H.T @ v


def flat_grad(params):
    return torch.cat([p.flatten() for p in params])


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


def collect_trajectories(env, policy: PolicyNet, n: int, max_steps: int = 500):
    trajs = []
    for _ in range(n):
        states, actions, env_rewards = [], [], []
        s, _ = env.reset()
        for _ in range(max_steps):
            a = policy.sample_action(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            states.append(torch.tensor(s, dtype=torch.float32))
            actions.append(torch.tensor(a))
            env_rewards.append(r)
            s = s_next
            if terminated or truncated:
                break
        trajs.append({"states": torch.stack(states), "actions": torch.stack(actions),
                      "env_rewards": env_rewards})
    return trajs


def train_ttsa(env, expert_trajs, config: dict, logger=None) -> dict:
    """TTSA bilevel оптимизация. Алгоритм не изменён."""
    train_cfg = config["training"]
    log_cfg = config.get("logging", {})

    n_iterations = int(train_cfg.get("n_outer_steps", 2000))
    n_traj_per_step = int(train_cfg.get("n_agent_traj", 20))
    alpha_inner = float(train_cfg.get("lr_inner", 3e-3))
    beta_outer = float(train_cfg.get("lr_outer", 3e-4))
    gamma = float(config.get("reward", {}).get("gamma", 0.99))
    metrics_every = int(log_cfg.get("log_every", 50))
    early_stop_len = 475

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim=state_dim, action_dim=action_dim)
    reward_net = RewardNet(state_dim=state_dim, gamma=gamma)
    ttsa = TTSANHD(policy, reward_net, n_cg_steps=10, reg=1e-2)

    inner_optimizer = torch.optim.SGD(policy.parameters(),     lr=alpha_inner)
    outer_optimizer = torch.optim.SGD(reward_net.parameters(), lr=beta_outer)


    history = {
        "l_outer": [], "l_inner": [], "env_reward": [],
        "hypgrad_norm": [], "agent_len": [],
    }

    for k in range(1, n_iterations + 1):
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
            l_out     = outer_loss(policy, expert_trajs).item()
            l_in      = inner_loss(policy, reward_net, agent_trajs).item()
            agent_len = float(np.mean([len(t["states"]) for t in agent_trajs]))
            env_r     = float(np.mean([sum(t["env_rewards"]) for t in agent_trajs]))

            history["l_outer"].append(l_out)
            history["l_inner"].append(l_in)
            history["hypgrad_norm"].append(hg_norm_before_clip)
            history["agent_len"].append(agent_len)
            history["env_reward"].append((k, env_r))

            row = (f"{k:>5} | {l_out:>10.3f} | {l_in:>10.3f} | "
                   f"{agent_len:>6.1f} | {env_r:>6.1f} | {hg_norm_before_clip:>8.4f}")
            logger.info(row) if logger else print(row)

            if agent_len >= early_stop_len:
                msg = f"[early stop] agent_len = {agent_len:.1f} >= {early_stop_len}"
                logger.info(msg) if logger else print(msg)
                break

    return history


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTSA IRL — CartPole")
    p.add_argument("--env", choices=["cartpole"], default="cartpole")
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
    logger  = get_logger("ttsa_cartpole", log_dir=log_dir)

    env = gym.make(config["env"]["id"])
    env.reset(seed=int(train_cfg.get("seed", 42)))

    # Экспертные траектории из data/ (не собираем заново)
    expert_train_path = data_cfg.get("expert_train_trajs", "data/cartpole/expert_train_trajs.pt")
    all_expert_trajs = torch.load(expert_train_path, map_location="cpu", weights_only=False)
    expert_trajs = all_expert_trajs
    logger.info(f"Загружено {len(expert_trajs)} экспертных траекторий из {expert_train_path}")

    logger.info("=== TTSA CartPole: начинаю оптимизацию ===")
    history = train_ttsa(env, expert_trajs, config, logger=logger)

    report_path = (
        Path(log_cfg.get("report_dir", "reports")) / "ttsa_cartpole_history.json"
    )
    save_history(history, str(report_path))
    logger.info(f"История сохранена в {report_path}")

    env.close()


if __name__ == "__main__":
    main()
