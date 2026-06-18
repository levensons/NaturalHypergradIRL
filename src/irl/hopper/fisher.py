"""
Fisher-NHD IRL with SAC inner agent for Hopper.

Usage:
    python -m src.irl.hopper.fisher --env hopper
    python -m src.irl.hopper.fisher --config configs/hopper.yaml
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import inner_loss, outer_loss, policy_nll, rank_corr
from src.agents.sac import SACInnerOptimizer
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import get_env_dims
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed, set_env_seed
from src.utils.torch import flat_grad, num_params, assign_flat_gradients
from src.utils.trajectories import (
    collect_trajectories,
    mean_trajectory_length,
    mean_trajectory_return,
)


class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

        self.init_weights()

    def init_weights(self):
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

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        self.init_weights()

    def init_weights(self):
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

        normalized_action = torch.clamp(
            self.action_to_normalized(actions),
            -1.0 + eps,
            1.0 - eps,
        )
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
        discounts = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float32, device=states.device),
            ts,
        )

        log_prob_sum = (discounts * self.policy.log_prob(states, actions)).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())

        return flat_grad(grads)

    def fisher(self, trajs) -> torch.Tensor:
        d = num_params(self.policy)
        fisher = torch.zeros(d, d, dtype=torch.float32)

        for traj in trajs:
            score = self.score(traj["states"], traj["actions"])
            fisher += torch.outer(score, score) / len(trajs)

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
            states = traj["states"]
            actions = traj["actions"]

            score_theta = self.score(states, actions)

            reward_sum = self.reward.trajectory_return(states, actions)
            grads_phi = torch.autograd.grad(reward_sum, self.reward.parameters())
            grad_phi = flat_grad(grads_phi)

            result += torch.dot(score_theta, v) * grad_phi

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
            cond_number = max_eig / max(min_eig, 1e-12)

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


def make_sac_inner_optimizer(
    sac_env,
    env,
    reward,
    policy,
    state_dim: int,
    action_dim: int,
    fisher_cfg: dict,
):
    inner_cfg = fisher_cfg["inner"]
    sac_cfg = inner_cfg["sac"]

    return SACInnerOptimizer(
        env=sac_env,
        reward=reward,
        policy=policy,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        lr_actor=float(inner_cfg["lr_actor"]),
        lr_q=float(sac_cfg["lr_q"]),
        buffer_size=int(sac_cfg["buffer_size"]),
        batch_size=int(sac_cfg["batch_size"]),
        learning_starts=int(sac_cfg["learning_starts"]),
        gamma=float(sac_cfg["gamma"]),
        tau=float(sac_cfg["tau"]),
        alpha=float(sac_cfg["alpha"]),
        autotune=bool(sac_cfg["autotune"]),
        policy_frequency=int(sac_cfg["policy_frequency"]),
        target_network_frequency=int(sac_cfg["target_network_frequency"]),
    )


def train_bilevel(env, expert_trajs, config: dict, logger=None) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]

    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    ckpt_cfg = config["checkpoint"]

    if inner_cfg["type"] != "sac":
        raise ValueError(f"Expected fisher.inner.type = sac, got {inner_cfg['type']}")

    expert_valid_trajs, random_valid_trajs = load_validation_trajectories(config)

    n_outer_steps = int(fisher_cfg["n_outer_steps"])
    n_inner_steps = int(fisher_cfg["n_inner_steps"])
    n_agent_traj = int(fisher_cfg["n_agent_traj"])

    state_dim, action_dim = get_env_dims(env)

    hidden = int(policy_cfg["hidden"])
    n_layers = int(policy_cfg["n_hidden_layers"])

    reward = Reward(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=int(reward_cfg["hidden"]),
        gamma=float(reward_cfg["gamma"]),
    )

    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden=hidden,
        n_hidden_layers=n_layers,
        log_std_min=float(policy_cfg["log_std_min"]),
        log_std_max=float(policy_cfg["log_std_max"]),
    )

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["reg"]),
        max_grad_norm=float(fisher_cfg["max_grad_norm"]),
        gamma=float(fisher_cfg["gamma"]),
    )

    sac_env = gym.make(config["env"]["id"])
    set_env_seed(sac_env, int(inner_cfg["sac_env_seed"]))

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

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "fisher_sac.pt")
    best_env_reward = float("-inf")

    arch = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "policy_hidden": hidden,
        "policy_n_hidden_layers": n_layers,
        "reward_hidden": int(reward_cfg["hidden"]),
        "reward_gamma": float(reward_cfg["gamma"]),
        "log_std_min": float(policy_cfg["log_std_min"]),
        "log_std_max": float(policy_cfg["log_std_max"]),
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "method": "fisher",
        "agent": "sac",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    def log_and_checkpoint(outer_step: int, agent_trajs):
        nonlocal best_env_reward

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss(policy, expert_trajs).item()

        agent_len = mean_trajectory_length(agent_trajs)
        expert_len = mean_trajectory_length(expert_trajs)

        agent_ret = mean_trajectory_return(agent_trajs)
        expert_ret = mean_trajectory_return(expert_trajs)

        rank_corr_val = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
        policy_nll_val = policy_nll(policy, expert_valid_trajs)

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

        if agent_ret > best_env_reward:
            best_env_reward = agent_ret

            save_checkpoint(
                path=best_checkpoint_path,
                policy=policy,
                reward=reward,
                arch=arch,
                outer_step=outer_step,
                best_env_reward=best_env_reward,
            )

        row = (
            f"{outer_step:>5} | {l_outer:>10.3f} | {agent_len:>10.1f} | "
            f"{expert_len:>10.1f} | {agent_ret:>10.1f} | {expert_ret:>10.1f} | "
            f"{rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )

        logger.info(row) if logger else print(row)

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )

    logger.info(header) if logger else print(header)

    try:
        inner_optimizer = make_sac_inner_optimizer(
            sac_env=sac_env,
            env=env,
            reward=reward,
            policy=policy,
            state_dim=state_dim,
            action_dim=action_dim,
            fisher_cfg=fisher_cfg,
        )

        inner_optimizer.optimize(
            n_inner_steps,
            inner_loss_fn=inner_loss,
            log_every=max(1, n_inner_steps // 10),
            n_log_traj=3,
        )

        agent_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_traj,
            max_steps=int(config["env"]["max_steps"]),
            desc="agent outer trajs",
        )

        log_and_checkpoint(outer_step=0, agent_trajs=agent_trajs)

        for outer_step in range(1, n_outer_steps + 1):
            outer_optimizer.step(expert_trajs, agent_trajs)

            inner_optimizer = make_sac_inner_optimizer(
                sac_env=sac_env,
                env=env,
                reward=reward,
                policy=policy,
                state_dim=state_dim,
                action_dim=action_dim,
                fisher_cfg=fisher_cfg,
            )

            inner_optimizer.optimize(
                n_inner_steps,
                inner_loss_fn=inner_loss,
                log_every=max(1, n_inner_steps // 10),
                n_log_traj=3,
            )

            agent_trajs = collect_trajectories(
                env=env,
                policy=policy,
                n=n_agent_traj,
                max_steps=int(config["env"]["max_steps"]),
                desc="agent outer trajs",
            )

            log_and_checkpoint(outer_step=outer_step, agent_trajs=agent_trajs)

    finally:
        sac_env.close()

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD IRL SAC — Hopper")

    parser.add_argument("--env", choices=["hopper"], default="hopper")
    parser.add_argument("--config", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)

    fisher_cfg = config["fisher"]
    log_cfg = config["logging"]

    set_random_seed(int(fisher_cfg["random_seed"]))

    log_dir = log_cfg["log_dir"]
    logger = get_logger("fisher_sac_hopper", log_dir=log_dir)

    env = gym.make(config["env"]["id"])
    set_env_seed(env, int(fisher_cfg["env_seed"]))

    try:
        expert_trajs, expert_train_path = load_expert_train_trajectories(config)
        logger.info(f"Loaded {len(expert_trajs)} expert trajectories from {expert_train_path}")

        logger.info("=== Fisher-NHD Hopper SAC ===")

        history = train_bilevel(
            env=env,
            expert_trajs=expert_trajs,
            config=config,
            logger=logger,
        )

        report_path = Path(log_cfg["report_dir"]) / "fisher_sac_hopper_history.json"
        save_history(history, str(report_path))

        logger.info(f"History saved to {report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
