"""
Fisher-NHD IRL for CartPole.

Usage:
    python -m src.irl.cartpole.fisher
    python -m src.irl.cartpole.fisher --config configs/cartpole.yaml
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from torch.distributions import Categorical
from tqdm import tqdm

from src.evaluation.metrics import inner_loss, outer_loss, policy_nll, rank_corr
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


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 16,
        n_hidden_layers: int = 1,
    ):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden), nn.Tanh()]

        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]

        layers.append(nn.Linear(hidden, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(states))

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution(states).log_prob(actions.long())

    def sample_action(self, state) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action = self.distribution(state_tensor).sample()

        return int(action.item())


class Reward(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 64,
        gamma: float = 0.99,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.net = nn.Linear(state_dim, 1, bias=True)
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(states).squeeze(-1)

    def discounted_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rewards = self.forward(states, actions)

        ts = torch.arange(
            states.size(0),
            dtype=torch.float32,
            device=states.device,
        )

        return torch.pow(self.gamma.to(states.device), ts) * rewards

    def trajectory_return(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()


class InnerOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        max_grad_norm: float,
        use_baseline: bool = False,
        normalize_coef: bool = False,
    ):
        self.reward = reward
        self.policy = policy
        self.max_grad_norm = max_grad_norm
        self.use_baseline = use_baseline
        self.normalize_coef = normalize_coef

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def grad(self, trajs) -> torch.Tensor:
        params = list(self.policy.parameters())
        gamma = float(self.reward.gamma.item())

        scores = []
        coefs = []

        for traj in trajs:
            states = traj["states"]
            actions = traj["actions"]

            T = states.size(0)

            gammas = torch.pow(
                torch.tensor(gamma, dtype=torch.float32, device=states.device),
                torch.arange(T, dtype=torch.float32, device=states.device),
            )

            log_probs = self.policy.log_prob(states, actions)
            log_prob_discounted = (gammas * log_probs).sum()

            with torch.no_grad():
                reward_return = self.reward.trajectory_return(states, actions)
                ell = (gammas * log_probs.detach()).sum() - reward_return

            score = flat_grad(
                torch.autograd.grad(
                    log_prob_discounted,
                    params,
                )
            )

            scores.append(score)
            coefs.append(ell)

        scores = torch.stack(scores)
        coefs = torch.stack(coefs).float()

        if self.use_baseline:
            coefs = coefs - coefs.mean()

        if self.normalize_coef:
            coefs = coefs / (coefs.std() + 1e-8)

        return (coefs.unsqueeze(1) * scores).mean(dim=0)

    def step(self, trajs) -> None:
        grad = self.grad(trajs)
        grad_norm = grad.norm().item()

        if grad_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / grad_norm)

        self.optimizer.zero_grad()
        assign_flat_gradients(self.policy, grad)
        self.optimizer.step()

    def optimize(
        self,
        env: Env,
        n_steps: int,
        n_traj: int,
        max_steps: int,
    ) -> None:
        for _ in tqdm(range(n_steps), desc="inner", leave=False):
            trajs = collect_trajectories(
                env=env,
                policy=self.policy,
                n=n_traj,
                max_steps=max_steps,
                desc="cartpole inner trajs",
                verbose=False,
            )
            self.step(trajs)


class OuterOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        max_grad_norm: float,
        fisher_reg: float,
        gamma: float,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.max_grad_norm = max_grad_norm

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

        self.optimizer = torch.optim.SGD(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def update_policy(self, policy: Policy) -> None:
        self.policy = policy

    def score(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())

        return flat_grad(grads)

    def fisher(self, trajs) -> torch.Tensor:
        d = num_params(self.policy)
        fisher = torch.zeros(d, d, dtype=torch.float32)

        for traj in trajs:
            score = self.score(traj["states"], traj["actions"])
            fisher += torch.outer(score, score)

        fisher /= len(trajs)
        fisher += self.fisher_reg * torch.eye(d, dtype=torch.float32)

        return fisher

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        d = num_params(self.policy)
        grad = torch.zeros(d, dtype=torch.float32)

        for traj in expert_trajs:
            grad += self.score(traj["states"], traj["actions"])

        return -grad / len(expert_trajs)

    def cross_derivative(self, trajs) -> torch.Tensor:
        d_theta = num_params(self.policy)
        d_phi = num_params(self.reward)

        cross = torch.zeros(d_theta, d_phi, dtype=torch.float32)

        for traj in trajs:
            states = traj["states"]
            actions = traj["actions"]

            score_theta = self.score(states, actions)

            reward_return = self.reward.trajectory_return(states, actions)
            grads_phi = torch.autograd.grad(
                reward_return,
                self.reward.parameters(),
            )
            grad_phi = flat_grad(grads_phi)

            cross += torch.outer(score_theta, grad_phi)

        return -cross / len(trajs)

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)
        outer_grad = self.outer_grad(expert_trajs)
        cross = self.cross_derivative(agent_trajs)

        fisher_inv_outer_grad = torch.linalg.solve(fisher, outer_grad)

        return -cross.T @ fisher_inv_outer_grad

    def step(self, expert_trajs, agent_trajs) -> torch.Tensor:
        hypergrad = self.hypergradient(expert_trajs, agent_trajs)

        self.raw_grad_norm = hypergrad.norm().item()

        if self.raw_grad_norm > self.max_grad_norm:
            hypergrad = hypergrad * (self.max_grad_norm / self.raw_grad_norm)

        self.clipped_grad_norm = hypergrad.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.reward, hypergrad)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return hypergrad


def train_bilevel(env: Env, config: dict, logger) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]

    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    ckpt_cfg = config["checkpoint"]

    if inner_cfg["type"] != "reinforce":
        raise ValueError(f"Expected fisher.inner.type = reinforce, got {inner_cfg['type']}")

    data_cfg = config["data"]

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    logger.info(f"Loaded {len(expert_train_trajs)} expert train trajectories from {expert_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} expert valid trajectories from {expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} random valid trajectories from {random_valid_path}")

    n_outer_steps = int(fisher_cfg["n_outer_steps"])
    n_inner_steps = int(fisher_cfg["n_inner_steps"])
    n_agent_traj = int(fisher_cfg["n_agent_traj"])

    max_steps = int(config["env"]["max_steps"])

    state_dim, action_dim = get_env_dims(env)

    hidden = int(policy_cfg["hidden"])
    n_hidden_layers = int(policy_cfg["n_hidden_layers"])

    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=hidden,
        n_hidden_layers=n_hidden_layers,
    )

    reward = Reward(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=int(reward_cfg["hidden"]),
        gamma=float(reward_cfg["gamma"]),
    )

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        max_grad_norm=float(fisher_cfg["max_grad_norm"]),
        gamma=float(fisher_cfg["gamma"]),
    )

    inner_optimizer = InnerOptimizer(
        reward=reward,
        policy=policy,
        lr=float(inner_cfg["lr_policy"]),
        max_grad_norm=float(inner_cfg["max_grad_norm"]),
        use_baseline=bool(inner_cfg["use_baseline"]),
        normalize_coef=bool(inner_cfg["normalize_coef"]),
    )

    history = {
        "l_outer": [],
        "l_inner": [],
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

    best_checkpoint_path = str(ckpt_dir / "fisher.pt")
    best_env_reward = float("-inf")

    arch = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "policy_hidden": hidden,
        "policy_n_hidden_layers": n_hidden_layers,
        "reward_hidden": int(reward_cfg["hidden"]),
        "reward_gamma": float(reward_cfg["gamma"]),
        "method": "fisher",
        "agent": "reinforce",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'L_inner':>10} | "
        f"{'agent_len':>10} | {'expert_len':>10} | "
        f"{'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | "
        f"{'hyp_raw':>10} | {'hyp_clip':>10} | {'lr_outer':>12}"
    )

    logger.info(header)

    for outer_step in range(1, n_outer_steps + 1):
        inner_optimizer.optimize(
            env=env,
            n_steps=n_inner_steps,
            n_traj=n_agent_traj,
            max_steps=max_steps,
        )

        agent_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_traj,
            max_steps=max_steps,
            desc="cartpole outer agent trajs",
            verbose=False,
        )

        outer_optimizer.step(expert_train_trajs, agent_trajs)

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss(policy, expert_train_trajs).item()
        l_inner = inner_loss(policy, reward, agent_trajs).item()

        agent_len = mean_trajectory_length(agent_trajs)
        expert_len = mean_trajectory_length(expert_train_trajs)

        agent_ret = mean_trajectory_return(agent_trajs)
        expert_ret = mean_trajectory_return(expert_train_trajs)

        rank_corr_val = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
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
            f"{outer_step:>5} | {l_outer:>10.3f} | {l_inner:>10.3f} | "
            f"{agent_len:>10.1f} | {expert_len:>10.1f} | "
            f"{agent_ret:>10.1f} | {expert_ret:>10.1f} | "
            f"{rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )

        logger.info(row)

    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD IRL — CartPole")

    parser.add_argument("--config", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("cartpole", args.config)
    config = load_config(config_path)

    fisher_cfg = config["fisher"]
    log_cfg = config["logging"]

    log_dir = log_cfg["log_dir"]
    logger = get_logger("fisher_cartpole", log_dir=log_dir)

    set_random_seed(int(fisher_cfg["random_seed"]))
    env = gym.make(config["env"]["id"])
    set_env_seed(env, int(fisher_cfg["env_seed"]))

    try:
        logger.info("=== Fisher-NHD CartPole ===")
        history = train_bilevel(env, config, logger)
        report_path = Path(log_cfg["report_dir"]) / "fisher_cartpole_history.json"
        save_history(history, str(report_path))
        logger.info(f"History saved to {report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
