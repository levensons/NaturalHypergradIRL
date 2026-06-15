"""
Fisher-NHD IRL с SAC-агентом для Hopper (непрерывная среда).

Запуск:
  python -m irl.hopper.fisher_sac_hopper --env hopper
  python -m irl.hopper.fisher_sac_hopper --config config/hopper.yaml
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from tqdm import tqdm

from evaluation.metrics import inner_loss, outer_loss, policy_nll, rank_corr
from src.agents.sac import SACInnerOptimizer
from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.config import load_config, resolve_config_path, set_seed, set_env_seed
from src.common.logging_utils import get_logger, save_history
from src.common.torch_utils import flat_grad, num_params, assign_flat_gradients, get_env_dims


class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        last_layer = self.net[-1]
        nn.init.uniform_(last_layer.weight, -1e-3, 1e-3)
        nn.init.zeros_(last_layer.bias)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1)).squeeze(-1)

    def discounted_rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        rewards = self.forward(states, actions)
        ts = torch.arange(states.size(0), dtype=torch.float32, device=states.device)
        return torch.pow(self.gamma.to(states.device), ts) * rewards

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        hidden: int = 64,
        n_hidden_layers: int = 2,
        log_std_max=2,
        log_std_min=-5,
    ):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, states: torch.Tensor):
        h = self.net(states)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, states: torch.Tensor, eps: float = 1e-3):
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1.0 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    @staticmethod
    def atanh(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def action_to_normalized(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_bias) / self.action_scale

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        mean, log_std = self.forward(states)
        std = log_std.exp()
        normalized_action = torch.clamp(self.action_to_normalized(actions), -1.0 + eps, 1.0 - eps)
        pre_tanh_action = self.atanh(normalized_action)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh_action)
        correction = torch.log(self.action_scale * (1.0 - normalized_action.pow(2)) + eps)
        return (log_prob - correction).sum(dim=-1)

    def sample_action(self, state, deterministic: bool = False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)
        with torch.no_grad():
            if deterministic:
                _, _, action = self.get_action(state_tensor)
            else:
                action, _, _ = self.get_action(state_tensor)
        return action.squeeze(0).cpu().numpy()


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


def collect_trajectories(env: Env, policy: Policy, n: int, max_steps=1000):
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

        trajs.append({"states": torch.stack(states), "actions": torch.stack(actions), "env_rewards": env_rewards})
    return trajs


class OuterOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        max_grad_norm=None,
        fisher_reg: float = 1.0,
        gamma: float = 0.99,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

        self.optimizer = torch.optim.SGD(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)

    def update_policy(self, policy: Policy):
        self.policy = policy

    def score(self, states, actions) -> torch.Tensor:
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def discounted_score(self, states, actions) -> torch.Tensor:
        ts = torch.arange(states.size(0), dtype=torch.float32, device=states.device)
        discounts = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=states.device), ts)
        log_prob_sum = (discounts * self.policy.log_prob(states, actions)).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def fisher(self, trajs) -> torch.Tensor:
        d = num_params(self.policy)
        fisher = torch.zeros(d, d, dtype=torch.float32)

        for traj in trajs:
            s = self.score(traj["states"], traj["actions"])
            fisher += torch.outer(s, s) / len(trajs)

        fisher += self.fisher_reg * torch.eye(d, dtype=torch.float32)
        return fisher

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        d = num_params(self.policy)
        grad = torch.zeros(d, dtype=torch.float32)

        for traj in expert_trajs:
            grad += self.discounted_score(traj["states"], traj["actions"])

        return -grad / len(expert_trajs)

    def cross_derivative_vec_product(self, trajs, v: torch.Tensor) -> torch.Tensor:
        d_phi = num_params(self.reward)
        result = torch.zeros(d_phi, dtype=torch.float32)

        for traj in trajs:
            states, actions = traj["states"], traj["actions"]
            s_theta = self.score(states, actions)
            reward_sum = self.reward.trajectory_return(states, actions)
            grads_phi = torch.autograd.grad(reward_sum, self.reward.parameters())
            g_phi = flat_grad(grads_phi)
            result += torch.dot(s_theta, v) * g_phi

        return -result / len(trajs)

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)
        outer_grad = self.outer_grad(expert_trajs)
        fisher_inv_outer_grad = torch.linalg.solve(fisher, outer_grad)
        hypergrad = self.cross_derivative_vec_product(agent_trajs, fisher_inv_outer_grad)

        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(fisher)
            min_eig = eigvals.min().item()
            max_eig = eigvals.max().item()
            cond_number = max_eig / min_eig

            print(
                f"Fisher stats | min_eig={min_eig:.3e} | max_eig={max_eig:.3e} | "
                f"cond={cond_number:.3e} | outer_grad_norm={outer_grad.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e}"
            )

        return hypergrad

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
    train_cfg = config["training"]
    inner_cfg = config.get("inner_agent", {})
    sac_cfg = inner_cfg.get("sac", {})
    outer_cfg = config.get("outer_optimizer", {})
    policy_cfg = config.get("policy", {})
    reward_cfg = config.get("reward", {})
    ckpt_cfg   = config.get("checkpoint", {})
    log_cfg    = config.get("logging", {})

    n_outer_steps = int(train_cfg["n_outer_steps"])
    n_inner_steps = int(train_cfg["n_inner_steps"])
    n_agent_traj = int(train_cfg["n_agent_traj"])

    state_dim, action_dim = get_env_dims(env)
    hidden = int(policy_cfg.get("hidden", 64))
    n_layers = int(policy_cfg.get("n_hidden_layers", 2))

    reward = Reward(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=int(reward_cfg.get("hidden", 64)),
        gamma=float(reward_cfg.get("gamma", 0.99)),
    )
    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden=hidden,
        n_hidden_layers=n_layers,
        log_std_min=float(policy_cfg.get("log_std_min", -5)),
        log_std_max=float(policy_cfg.get("log_std_max", 2)),
    )

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=float(train_cfg["lr_outer"]),
        fisher_reg=float(outer_cfg.get("fisher_reg", 1.0)),
        max_grad_norm=outer_cfg.get("max_grad_norm"),
        gamma=float(outer_cfg.get("gamma", 0.99)),
    )

    sac_env = gym.make(config["env"]["id"])
    set_env_seed(sac_env, int(train_cfg.get("seed", 42)) + 1)

    inner_optimizer = SACInnerOptimizer(
        env=sac_env,
        reward=reward,
        policy=policy,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        lr_actor=float(train_cfg["lr_inner"]),
        lr_q=float(sac_cfg.get("lr_q", 1e-3)),
        buffer_size=int(sac_cfg.get("buffer_size", 300_000)),
        batch_size=int(sac_cfg.get("batch_size", 256)),
        learning_starts=int(sac_cfg.get("learning_starts", 500)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        alpha=float(sac_cfg.get("alpha", 0.2)),
        autotune=bool(sac_cfg.get("autotune", True)),
        policy_frequency=int(sac_cfg.get("policy_frequency", 2)),
        target_network_frequency=int(sac_cfg.get("target_network_frequency", 1)),
    )

    history = {
        "l_outer": [],
        "agent_len": [],
        "expert_len": [],
        "agent_return": [],
        "expert_return": [],
        "rank_corr": [],
        "policy_nll": [],
        "raw_hypgrad_norm": [],
        "clipped_hypgrad_norm": [],
        "lr_outer": [],
    }

    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/hopper"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = str(ckpt_dir / "fisher_sac.pt")
    best_l_outer = float("inf")

    arch = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "policy_hidden": hidden,
        "policy_n_hidden_layers": n_layers,
        "reward_hidden": int(reward_cfg.get("hidden", 64)),
        "reward_gamma": float(reward_cfg.get("gamma", 0.99)),
        "log_std_min": float(policy_cfg.get("log_std_min", -5)),
        "log_std_max": float(policy_cfg.get("log_std_max", 2)),
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "method": "fisher",
        "agent": "sac",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )
    if logger:
        logger.info(header)
    else:
        print(header)

    for outer_step in range(1, n_outer_steps + 1):
        inner_optimizer.optimize(
            n_inner_steps,
            inner_loss_fn=inner_loss,
            log_every=int(log_cfg.get("inner_log_every", 10000)),
            n_log_traj=3,
        )
        agent_trajs = collect_trajectories(env, policy, n_agent_traj)
        outer_optimizer.step(expert_trajs, agent_trajs)

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss(policy, expert_trajs).item()
        agent_len = np.mean([len(t["states"]) for t in agent_trajs])
        expert_len = np.mean([len(t["states"]) for t in expert_trajs])
        agent_ret = np.mean([sum(t["env_rewards"]) for t in agent_trajs])
        expert_ret = np.mean([sum(t["env_rewards"]) for t in expert_trajs])
        rank_corr_val = rank_corr(reward, expert_trajs + agent_trajs)
        policy_nll_val = policy_nll(policy, expert_trajs)

        history["l_outer"].append(l_outer)
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
                path=best_checkpoint_path,
                policy=policy,
                reward=reward,
                arch=arch,
                outer_step=outer_step,
                best_l_outer=best_l_outer,
            )

        row = (
            f"{outer_step:>5} | {l_outer:>10.3f} | {agent_len:>10.1f} | "
            f"{expert_len:>10.1f} | {agent_ret:>10.1f} | {expert_ret:>10.1f} | "
            f"{rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | {raw_hypgrad_norm:>10.3f} | "
            f"{clipped_hypgrad_norm:>10.3f} | {lr_outer_current:>12.2e}"
        )
        if logger:
            logger.info(row)
        else:
            print(row)

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )
    if logger:
        logger.info(header)
    else:
        print(header)

    inner_optimizer.optimize(n_inner_steps)
    agent_trajs = collect_trajectories(env, policy, n_agent_traj)
    log_and_checkpoint(outer_step=0, agent_trajs=agent_trajs)

    for outer_step in range(1, n_outer_steps + 1):
        outer_optimizer.step(expert_trajs, agent_trajs)

        inner_optimizer = SACInnerOptimizer(
            env=sac_env,
            reward=reward,
            policy=policy,
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            lr_actor=float(train_cfg["lr_inner"]),
            lr_q=float(sac_cfg.get("lr_q", 1e-3)),
            buffer_size=int(sac_cfg.get("buffer_size", 300_000)),
            batch_size=int(sac_cfg.get("batch_size", 256)),
            learning_starts=int(sac_cfg.get("learning_starts", 500)),
            gamma=float(sac_cfg.get("gamma", 0.99)),
            tau=float(sac_cfg.get("tau", 0.005)),
            alpha=float(sac_cfg.get("alpha", 0.2)),
            autotune=bool(sac_cfg.get("autotune", True)),
            policy_frequency=int(sac_cfg.get("policy_frequency", 2)),
            target_network_frequency=int(sac_cfg.get("target_network_frequency", 1)),
        )

        inner_optimizer.optimize(n_inner_steps)
        agent_trajs = collect_trajectories(env, policy, n_agent_traj)
        log_and_checkpoint(outer_step=outer_step, agent_trajs=agent_trajs)

    sac_env.close()
    return history


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fisher-NHD IRL SAC — Hopper")
    p.add_argument("--env", choices=["hopper"], default="hopper")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse()
    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)

    train_cfg = config["training"]
    log_cfg = config.get("logging", {})
    data_cfg = config.get("data", {})

    set_seed(int(train_cfg.get("seed", 42)))

    log_dir = log_cfg.get("log_dir", "logs")
    logger = get_logger("fisher_sac_hopper", log_dir=log_dir)

    env = gym.make(config["env"]["id"])
    set_env_seed(env, int(train_cfg.get("seed", 42)))

    expert_train_path = data_cfg.get("expert_train_trajs", "data/hopper/expert_train_trajs.pt")
    all_expert_trajs = torch.load(expert_train_path, map_location="cpu", weights_only=False)

    expert_trajs = all_expert_trajs
    logger.info(f"Загружено {len(expert_trajs)} экспертных траекторий из {expert_train_path}")

    logger.info("=== Fisher-NHD Hopper SAC ===")
    history = train_bilevel(env, expert_trajs, config, logger=logger)

    report_path = Path(log_cfg.get("report_dir", "reports")) / "fisher_sac_hopper_history.json"
    save_history(history, str(report_path))
    logger.info(f"История сохранена в {report_path}")

    env.close()


if __name__ == "__main__":
    main()
