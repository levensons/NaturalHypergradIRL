"""
TTSA baseline IRL for CartPole.

Usage:
    python -m src.irl.cartpole.ttsa
    python -m src.irl.cartpole.ttsa --config configs/cartpole.yaml
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.evaluation.metrics import inner_loss, outer_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import get_env_dims
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed, set_env_seed
from src.utils.torch import flat_grad, assign_flat_gradients, safe_clip_grad
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 16,
        n_hidden_layers: int = 1,
    ):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden), nn.ReLU()]

        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]

        layers.append(nn.Linear(hidden, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(states))

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution(states).log_prob(actions.long())

    def sample_action(self, state) -> int:
        state_t = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action = self.distribution(state_t).sample()

        return int(action.item())


class Reward(nn.Module):
    def __init__(self, state_dim: int, gamma: float = 0.99):
        super().__init__()

        self.net = nn.Linear(state_dim, 1, bias=True)
        self.gamma = gamma

    def base_reward(
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
        ts = torch.arange(
            states.size(0),
            dtype=torch.float32,
            device=states.device,
        )

        return self.base_reward(states, actions) * (self.gamma**ts)

    def discounted_return(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()

    def trajectory_return(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.discounted_return(states, actions)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_reward(states, actions)


class TTSANHD(nn.Module):
    def __init__(
        self,
        policy: Policy,
        reward: Reward,
        n_cg_steps: int,
        fisher_reg: float,
    ):
        super().__init__()

        self.policy = policy
        self.reward = reward
        self.n_cg_steps = n_cg_steps
        self.fisher_reg = fisher_reg

    def score(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())

        return flat_grad(grads)

    def reinforce_grad(self, trajs) -> torch.Tensor:
        d = sum(p.numel() for p in self.policy.parameters())

        losses = []

        for traj in trajs:
            states = traj["states"]
            actions = traj["actions"]
            T = len(states)

            gammas = torch.tensor(
                [self.reward.gamma**t for t in range(T)],
                dtype=torch.float32,
                device=states.device,
            )

            with torch.no_grad():
                loss = (gammas * (self.policy.log_prob(states, actions) - self.reward(states, actions))).sum().item()

            losses.append(loss)

        baseline = float(np.mean(losses))
        grad = torch.zeros(d, dtype=torch.float32)

        for traj, loss in zip(trajs, losses):
            states = traj["states"]
            actions = traj["actions"]

            grad += (loss - baseline) * self.score(states, actions)

        return grad / len(trajs)

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        d = sum(p.numel() for p in self.policy.parameters())
        grad = torch.zeros(d, dtype=torch.float32)

        for traj in expert_trajs:
            grad += self.score(traj["states"], traj["actions"])

        return -grad / len(expert_trajs)

    def cross_derivative(self, trajs) -> torch.Tensor:
        d_theta = sum(p.numel() for p in self.policy.parameters())
        d_phi = sum(p.numel() for p in self.reward.parameters())

        cross = torch.zeros(d_theta, d_phi, dtype=torch.float32)

        for traj in trajs:
            states = traj["states"]
            actions = traj["actions"]

            score_theta = self.score(states, actions)

            reward_sum = self.reward.discounted_return(states, actions)
            grads_phi = torch.autograd.grad(reward_sum, self.reward.parameters())
            grad_phi = flat_grad(grads_phi)

            cross += torch.outer(score_theta, grad_phi)

        return -cross / len(trajs)

    def fisher_vector_product(self, trajs, vector: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(vector)

        for traj in trajs:
            score_theta = self.score(traj["states"], traj["actions"])
            result += score_theta * (score_theta @ vector)

        result /= len(trajs)

        return result + self.fisher_reg * vector

    def conjugate_gradient_solve(
        self,
        trajs,
        grad: torch.Tensor,
        tol: float = 1e-8,
    ) -> torch.Tensor:
        solution = torch.zeros_like(grad)

        residual = grad.clone()
        direction = residual.clone()
        residual_dot = (residual * residual).sum()

        if residual_dot < tol:
            return solution

        for _ in range(self.n_cg_steps):
            fisher_direction = self.fisher_vector_product(trajs, direction)
            curvature = (direction * fisher_direction).sum()

            if curvature <= 0:
                break

            alpha = residual_dot / curvature

            solution = solution + alpha * direction
            residual = residual - alpha * fisher_direction

            new_residual_dot = (residual * residual).sum()

            if new_residual_dot < tol:
                break

            direction = residual + (new_residual_dot / residual_dot) * direction
            residual_dot = new_residual_dot

        return solution

    def forward(self, expert_trajs, agent_trajs) -> torch.Tensor:
        outer_grad = self.outer_grad(expert_trajs)

        fisher_inv_outer_grad = self.conjugate_gradient_solve(
            agent_trajs,
            outer_grad,
        )

        cross = self.cross_derivative(agent_trajs)

        return -cross.T @ fisher_inv_outer_grad


def train_ttsa(env, config: dict, logger) -> dict:
    ttsa_cfg = config["ttsa"]
    policy_cfg = config["policy"]
    ckpt_cfg = config["checkpoint"]

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

    n_iterations = int(ttsa_cfg["n_iterations"])
    n_traj_per_step = int(ttsa_cfg["n_traj_per_step"])
    metrics_every = int(ttsa_cfg["metrics_every"])

    lr_policy = float(ttsa_cfg["lr_policy"])
    lr_reward = float(ttsa_cfg["lr_reward"])

    gamma = float(ttsa_cfg["gamma"])
    n_cg_steps = int(ttsa_cfg["n_cg_steps"])
    fisher_reg = float(ttsa_cfg["fisher_reg"])

    inner_grad_max_norm = float(ttsa_cfg["inner_grad_max_norm"])
    outer_grad_max_norm = float(ttsa_cfg["outer_grad_max_norm"])

    early_stop_len = float(
        ttsa_cfg.get(
            "early_stop_len",
            config["env"]["max_steps"],
        )
    )

    state_dim, action_dim = get_env_dims(env)

    hidden = int(policy_cfg["hidden"])
    n_hidden_layers = int(policy_cfg["n_hidden_layers"])

    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=hidden,
        n_hidden_layers=n_hidden_layers,
    )

    reward_net = Reward(
        state_dim=state_dim,
        gamma=gamma,
    )

    ttsa = TTSANHD(
        policy=policy,
        reward=reward_net,
        n_cg_steps=n_cg_steps,
        fisher_reg=fisher_reg,
    )

    inner_optimizer = torch.optim.SGD(policy.parameters(), lr=lr_policy)
    outer_optimizer = torch.optim.SGD(reward_net.parameters(), lr=lr_reward)

    history = {
        "l_outer": [],
        "l_inner": [],
        "env_reward": [],
        "hypgrad_norm": [],
        "agent_len": [],
        "iter_time": [],
    }

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "ttsa.pt")
    best_env_reward = float("-inf")

    arch = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "policy_hidden": hidden,
        "policy_n_hidden_layers": n_hidden_layers,
        "reward_gamma": gamma,
        "method": "ttsa",
        "agent": "reinforce",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    logger.info("Step |    L_outer |    L_inner |    len |   EnvR |    |hyp| | t/iter")
    logger.info("-" * 78)

    for iteration in range(1, n_iterations + 1):
        t_iter = time.time()

        agent_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_traj_per_step,
            max_steps=int(config["env"]["max_steps"]),
            desc="ttsa cartpole agent trajs",
            verbose=False,
        )

        inner_grad, ok_inner = safe_clip_grad(
            ttsa.reinforce_grad(agent_trajs),
            max_norm=inner_grad_max_norm,
        )

        if not ok_inner:
            msg = f"[iter {iteration}] WARNING: NaN/Inf in inner_grad, skipping inner step"
            logger.warning(msg)
        else:
            inner_optimizer.zero_grad()
            assign_flat_gradients(policy, inner_grad)
            inner_optimizer.step()

        hypergrad = ttsa(expert_train_trajs, agent_trajs)

        hypgrad_norm_before_clip = hypergrad.norm().item() if torch.isfinite(hypergrad).all() else float("inf")

        hypergrad, ok_outer = safe_clip_grad(
            hypergrad,
            max_norm=outer_grad_max_norm,
        )

        if not ok_outer:
            msg = f"[iter {iteration}] WARNING: NaN/Inf in hypergrad, skipping outer step"
            logger.warning(msg)
        else:
            outer_optimizer.zero_grad()
            assign_flat_gradients(reward_net, hypergrad)
            outer_optimizer.step()

        if iteration % metrics_every == 0 or iteration == 1:
            l_out = outer_loss(policy, expert_train_trajs).item()
            l_in = inner_loss(policy, reward_net, agent_trajs).item()

            agent_len = mean_trajectory_length(agent_trajs)
            env_r = mean_trajectory_return(agent_trajs)

            iter_elapsed = time.time() - t_iter

            history["l_outer"].append(l_out)
            history["l_inner"].append(l_in)
            history["hypgrad_norm"].append(hypgrad_norm_before_clip)
            history["agent_len"].append(agent_len)
            history["env_reward"].append((iteration, env_r))
            history["iter_time"].append(iter_elapsed)

            if env_r > best_env_reward:
                best_env_reward = env_r

                save_checkpoint(
                    path=best_checkpoint_path,
                    policy=policy,
                    reward=reward_net,
                    arch=arch,
                    iteration=iteration,
                    best_env_reward=best_env_reward,
                )

            row = (
                f"{iteration:>5} | {l_out:>10.3f} | {l_in:>10.3f} | "
                f"{agent_len:>6.1f} | {env_r:>6.1f} | "
                f"{hypgrad_norm_before_clip:>8.4f} | {iter_elapsed:>6.1f}s"
            )

            logger.info(row)

            if agent_len >= early_stop_len:
                msg = f"[early stop] agent_len = {agent_len:.1f} " f">= {early_stop_len:.1f}"
                logger.info(msg)
                break

    total_time = sum(history["iter_time"])
    msg = f"Total logged iteration time: {total_time:.1f}s ({total_time / 60:.1f} min)"
    logger.info(msg)

    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTSA IRL — CartPole")

    parser.add_argument("--config", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("cartpole", args.config)
    config = load_config(config_path)

    ttsa_cfg = config["ttsa"]
    log_cfg = config["logging"]

    set_random_seed(int(ttsa_cfg["random_seed"]))

    log_dir = log_cfg["log_dir"]
    logger = get_logger("ttsa_cartpole", log_dir=log_dir)

    env = gym.make(config["env"]["id"])
    set_env_seed(env, int(ttsa_cfg["env_seed"]))

    try:
        logger.info("=== TTSA CartPole ===")
        history = train_ttsa(env, config, logger)
        report_path = Path(log_cfg["report_dir"]) / "ttsa_cartpole_history.json"
        save_history(history, str(report_path))
        logger.info(f"History saved to {report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
