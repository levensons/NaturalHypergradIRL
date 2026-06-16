"""
Fisher-NHD IRL с REINFORCE-агентом для Hopper (непрерывная среда).

Запуск:
  python -m irl.hopper.fisher_reinforce_hopper --env hopper
  python -m irl.hopper.fisher_reinforce_hopper --config config/hopper.yaml
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from torch.distributions import Normal
from tqdm import tqdm

from evaluation.metrics import inner_loss, outer_loss, policy_nll, rank_corr
from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.config import load_config, resolve_config_path, set_seed
from src.common.logging_utils import get_logger, save_history


def flat_grad(params):
    return torch.cat([p.flatten() for p in params])


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def assign_flat_gradients(module, flat_grad: torch.Tensor):
    i = 0
    for p in module.parameters():
        n = p.numel()
        grad_chunk = flat_grad[i : i + n]
        if grad_chunk.numel() != n:
            raise ValueError("Flat gradient has incorrect size: not enough elements.")
        p.grad = grad_chunk.reshape(p.shape).clone()
        i += n
    if i != flat_grad.numel():
        raise ValueError("Flat gradient has incorrect size: too many elements.")


def get_env_dims(env: Env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return state_dim, action_dim


class Policy(nn.Module):
    """Gaussian policy с параметрическим log_std. n_hidden_layers=2 соответствует чекпоинту."""

    def __init__(self, state_dim, action_dim, hidden=64, n_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def distribution(self, states):
        mu, std = self.forward(states)
        return Normal(mu, std)

    def log_prob(self, states, actions):
        return self.distribution(states).log_prob(actions).sum(-1)

    def sample_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            dist = self.distribution(state_tensor)
            action = dist.sample()
        return action.numpy()


class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([states, actions], dim=-1)
        return self.net(sa).squeeze(-1)

    def discounted_rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        rewards = self.forward(states, actions)
        ts = torch.arange(states.size(0), dtype=torch.float32)
        return torch.pow(self.gamma, ts) * rewards

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()


class SB3PolicyWrapper:
    def __init__(self, model, action_space):
        self.model = model
        self.action_low = action_space.low
        self.action_high = action_space.high

    def sample_action(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return np.clip(action, self.action_low, self.action_high)

    def eval(self):
        pass


def collect_trajectories(env: Env, policy, n: int, max_steps=1000):
    trajs = []
    for _ in tqdm(range(n), desc="collect trajs", leave=False):
        states, actions, env_rewards = [], [], []
        s, _ = env.reset()
        for _ in range(max_steps):
            a = policy.sample_action(s)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            s_next, r, terminated, truncated, _ = env.step(a)
            states.append(torch.tensor(s, dtype=torch.float32))
            actions.append(torch.tensor(a, dtype=torch.float32))
            env_rewards.append(r)
            s = s_next
            if terminated or truncated:
                break
        trajs.append({"states": torch.stack(states), "actions": torch.stack(actions),
                      "env_rewards": env_rewards})
    return trajs


class InnerOptimizer:
    def __init__(self, reward: Reward, policy: Policy, lr: float, max_grad_norm: float,
                 use_baseline: bool = False, normalize_coef: bool = False, eta_min: float = 0.0):
        self.policy = policy
        self.reward = reward
        self.use_baseline = use_baseline
        self.normalize_coef = normalize_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.eta_min = eta_min
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr)
        self.scheduler = None

    def grad(self, trajs) -> torch.Tensor:
        params = list(self.policy.parameters())
        scores, coefs = [], []
        for traj in trajs:
            states, actions = traj["states"], traj["actions"]
            log_prob_sum = self.policy.log_prob(states, actions).sum()
            with torch.no_grad():
                reward_return = self.reward.trajectory_return(states, actions)
                ell = log_prob_sum.detach() - reward_return
                coef = ell + 1.0
            score = flat_grad(torch.autograd.grad(log_prob_sum, params))
            scores.append(score)
            coefs.append(coef)
        scores = torch.stack(scores)
        coefs = torch.stack(coefs).float()
        if self.use_baseline:
            coefs = coefs - coefs.mean()
        if self.normalize_coef:
            coefs = coefs / (coefs.std() + 1e-8)
        return (coefs.unsqueeze(1) * scores).mean(dim=0)

    def step(self, trajs):
        grad = self.grad(trajs)
        raw_norm = grad.norm().item()
        if raw_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / raw_norm)
        self.optimizer.zero_grad()
        assign_flat_gradients(self.policy, grad)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return {"inner_grad_raw": raw_norm, "inner_grad_clip": grad.norm().item()}

    def optimize(self, env, n_steps: int, n_traj: int):
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, n_steps, eta_min=self.eta_min)
        pbar = tqdm(range(n_steps), desc="REINFORCE inner", leave=False)
        for _ in pbar:
            trajs = collect_trajectories(env, self.policy, n=n_traj)
            stats = self.step(trajs)
            pbar.set_postfix({"g_raw": f"{stats['inner_grad_raw']:.2e}",
                              "g_clip": f"{stats['inner_grad_clip']:.2e}"})


class OuterOptimizer:
    def __init__(self, reward: Reward, policy, lr: float, max_grad_norm,
                 fisher_reg: float = 1e-2, gamma: float = 0.99):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.max_grad_norm = max_grad_norm
        self.raw_grad_norm = None
        self.clipped_grad_norm = None
        self.optimizer = torch.optim.SGD(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)

    def update_policy(self, policy):
        self.policy = policy

    def score(self, states, actions) -> torch.Tensor:
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def fisher(self, trajs) -> torch.Tensor:
        d = num_params(self.policy)
        F = torch.zeros(d, d)
        for traj in trajs:
            s = self.score(traj["states"], traj["actions"])
            F += torch.outer(s, s)
        F /= len(trajs)
        F += self.fisher_reg * torch.eye(d)
        return F

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        d = num_params(self.policy)
        grad = torch.zeros(d)
        for traj in expert_trajs:
            grad += self.score(traj["states"], traj["actions"])
        return -grad / len(expert_trajs)

    def cross_derivative(self, trajs) -> torch.Tensor:
        d_theta = num_params(self.policy)
        d_phi = num_params(self.reward)
        H = torch.zeros(d_theta, d_phi)
        for traj in trajs:
            states, actions = traj["states"], traj["actions"]
            s_theta = self.score(states, actions)
            reward_sum = self.reward.trajectory_return(states, actions)
            grads_phi = torch.autograd.grad(reward_sum, self.reward.parameters())
            g_phi = flat_grad(grads_phi)
            H += torch.outer(s_theta, g_phi)
        return -H / len(trajs)

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)
        outer_grad = self.outer_grad(expert_trajs)
        cross = self.cross_derivative(agent_trajs)
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(fisher)
            min_eig = eigvals.min().item()
            max_eig = eigvals.max().item()
            cond_number = max_eig / max(min_eig, 1e-12)
            print(f"Fisher stats | min_eig={min_eig:.3e} | max_eig={max_eig:.3e} | "
                  f"cond={cond_number:.3e} | outer_grad_norm={outer_grad.norm().item():.3e} | "
                  f"cross_norm={cross.norm().item():.3e}")
        fisher_inv_outer_grad = torch.linalg.solve(fisher, outer_grad)
        return -torch.einsum("tp,t->p", cross, fisher_inv_outer_grad)

    def step(self, expert_trajs, agent_trajs) -> torch.Tensor:
        hypergrad = self.hypergradient(expert_trajs, agent_trajs)
        self.raw_grad_norm = hypergrad.norm().item()
        if self.max_grad_norm is not None and self.raw_grad_norm > self.max_grad_norm:
            hypergrad = hypergrad * (self.max_grad_norm / self.raw_grad_norm)
        self.clipped_grad_norm = hypergrad.norm().item()
        self.optimizer.zero_grad()
        assign_flat_gradients(self.reward, hypergrad)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return hypergrad


def train_bilevel(env, expert_trajs, config: dict, logger=None) -> dict:
    """Двухуровневая оптимизация Fisher-NHD REINFORCE. Алгоритм не изменён."""
    train_cfg  = config["training"]
    inner_cfg  = config.get("inner_agent", {})
    outer_cfg  = config.get("outer_optimizer", {})
    policy_cfg = config.get("policy", {})
    reward_cfg = config.get("reward", {})
    ckpt_cfg   = config.get("checkpoint", {})
    data_cfg   = config.get("data", {})

    expert_valid_trajs = torch.load(
        data_cfg.get("expert_valid_trajs", "data/hopper/expert_valid_trajs.pt"),
        map_location="cpu", weights_only=False,
    )
    random_valid_trajs = torch.load(
        data_cfg.get("random_valid_trajs", "data/hopper/random_valid_trajs.pt"),
        map_location="cpu", weights_only=False,
    )

    n_outer_steps      = int(train_cfg["n_outer_steps"])
    n_inner_steps      = int(train_cfg["n_inner_steps"])
    n_agent_inner_traj = int(train_cfg["n_agent_traj"])
    n_agent_outer_traj = int(train_cfg["n_agent_traj"])

    state_dim, action_dim = get_env_dims(env)

    hidden   = int(policy_cfg.get("hidden", 64))
    n_layers = int(policy_cfg.get("n_hidden_layers", 2))

    reward = Reward(
        state_dim=state_dim, action_dim=action_dim,
        hidden=int(reward_cfg.get("hidden", 64)),
        gamma=float(reward_cfg.get("gamma", 0.99)),
    )

    outer_optimizer = OuterOptimizer(
        reward=reward, policy=None,
        lr=float(train_cfg["lr_outer"]),
        fisher_reg=float(outer_cfg.get("fisher_reg", 0.01)),
        max_grad_norm=outer_cfg.get("max_grad_norm"),
        gamma=float(outer_cfg.get("gamma", 0.99)),
    )


    history = {
        "l_outer": [], "l_inner": [], "agent_len": [], "expert_len": [],
        "agent_return": [], "expert_return": [], "rank_corr": [], "policy_nll": [],
        "raw_hypgrad_norm": [], "clipped_hypgrad_norm": [], "lr_outer": [],
    }

    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/hopper"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = str(ckpt_dir / "fisher_reinforce.pt")
    best_l_outer = float("inf")

    arch = {
        "state_dim": state_dim, "action_dim": action_dim,
        "policy_hidden": hidden, "policy_n_hidden_layers": n_layers,
        "reward_hidden": int(reward_cfg.get("hidden", 64)),
        "reward_gamma": float(reward_cfg.get("gamma", 0.99)),
        "method": "fisher", "agent": "reinforce",
        "env_name": config["env"]["name"], "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'L_inner':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )
    if logger:
        logger.info(header)
    else:
        print(header)

    for outer_step in range(1, n_outer_steps + 1):
        policy = Policy(state_dim=state_dim, action_dim=action_dim,
                        hidden=hidden, n_hidden_layers=n_layers)
        inner_optimizer = InnerOptimizer(
            reward=reward, policy=policy,
            lr=float(train_cfg["lr_inner"]),
            use_baseline=bool(inner_cfg.get("use_baseline", True)),
            normalize_coef=bool(inner_cfg.get("normalize_coef", True)),
            max_grad_norm=float(inner_cfg.get("max_grad_norm", 1.0)),
            eta_min=1e-6,
        )
        inner_optimizer.optimize(env, n_inner_steps, n_agent_inner_traj)

        agent_trajs = collect_trajectories(env, policy, n_agent_outer_traj)
        outer_optimizer.update_policy(policy)
        outer_optimizer.step(expert_trajs, agent_trajs)

        lr_outer_current     = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm     = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer    = outer_loss(policy, expert_trajs).item()
        l_inner    = inner_loss(policy, reward, agent_trajs).item()
        agent_len  = np.mean([len(t["states"]) for t in agent_trajs])
        expert_len = np.mean([len(t["states"]) for t in expert_trajs])
        agent_ret  = np.mean([sum(t["env_rewards"]) for t in agent_trajs])
        expert_ret = np.mean([sum(t["env_rewards"]) for t in expert_trajs])
        rank_corr_val  = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
        policy_nll_val = policy_nll(policy, expert_valid_trajs)

        history["l_outer"].append(l_outer)
        history["l_inner"].append(l_inner)
        history["agent_len"].append(agent_len)
        history["expert_len"].append(expert_len)
        history["agent_return"].append(agent_ret)
        history["expert_return"].append(expert_ret)
        history["rank_corr"].append(rank_corr_val)
        history["policy_nll"].append(policy_nll_val)
        history["raw_hypgrad_norm"].append(raw_hypgrad_norm)
        history["clipped_hypgrad_norm"].append(clipped_hypgrad_norm)
        history["lr_outer"].append(lr_outer_current)

        if l_outer < best_l_outer:
            best_l_outer = l_outer
            save_checkpoint(
                path=best_checkpoint_path, policy=policy, reward=reward, arch=arch,
                outer_step=outer_step, best_l_outer=best_l_outer,
            )

        row = (
            f"{outer_step:>5} | {l_outer:>10.3f} | {l_inner:>10.3f} | "
            f"{agent_len:>10.1f} | {expert_len:>10.1f} | {agent_ret:>10.1f} | "
            f"{expert_ret:>10.1f} | {rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )
        if logger:
            logger.info(row)
        else:
            print(row)

    return history


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fisher-NHD IRL REINFORCE — Hopper")
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
    logger  = get_logger("fisher_reinforce_hopper", log_dir=log_dir)

    env = gym.make(config["env"]["id"])
    env.reset(seed=int(train_cfg.get("seed", 42)))

    expert_train_path = data_cfg.get("expert_train_trajs", "data/hopper/expert_train_trajs.pt")
    all_expert_trajs = torch.load(expert_train_path, map_location="cpu", weights_only=False)

    expert_trajs = all_expert_trajs
    logger.info(f"Загружено {len(expert_trajs)} экспертных траекторий из {expert_train_path}")

    logger.info("=== Fisher-NHD Hopper REINFORCE ===")
    history = train_bilevel(env, expert_trajs, config, logger=logger)

    report_path = (
        Path(log_cfg.get("report_dir", "reports")) / "fisher_reinforce_hopper_history.json"
    )
    save_history(history, str(report_path))
    logger.info(f"История сохранена в {report_path}")

    env.close()


if __name__ == "__main__":
    main()
