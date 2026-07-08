"""
Fisher-NHD IRL with SAC inner agent for Hopper.

Usage:
    python -m src.irl.hopper.fisher
    python -m src.irl.hopper.fisher --config configs/hopper.yaml
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import mlflow

import gymnasium as gym
from gymnasium import Env

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import outer_loss, policy_nll, rank_corr, learned_reward_stats, inner_loss
from src.agents.sac import SAC
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import Environment
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed
from src.utils.torch import flat_grad, num_params, assign_flat_gradients, to_device
from src.utils.trajectories import (
    collect_trajectories,
    mean_trajectory_length,
    mean_trajectory_return,
    discount_weights,
)


class Reward(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64,
        clamp_magnitude: float = 10.0,
    ):
        super().__init__()

        self.clamp_magnitude = clamp_magnitude

        layers = []
        in_dim = state_dim + action_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([states, actions], dim=-1))
        out = out.squeeze(-1)
        out = torch.clamp(out, -self.clamp_magnitude, self.clamp_magnitude)
        return out # (B,)

    def rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions) # (B,)

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions).sum()
    
    def as_fn(self):
        def reward_fn(states: torch.Tensor, actions: torch.Tensor):
            was_training = self.training
            if was_training:
                self.eval()

            with torch.no_grad():
                out = self.rewards(states, actions)

            if was_training:
                self.train()

            return out # (B,)
        
        return reward_fn


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        hidden_dim: int = 64,
        n_hidden_layers: int = 1,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()

        in_dim = state_dim
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

        self.register_buffer(
            "action_low", torch.as_tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(action_high, dtype=torch.float32)
        )
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, states: torch.Tensor):
        x = self.backbone(states)
        mean = self.mean_head(x)  # (B, action_dim)
        log_std = self.log_std_head(x)  # (B, action_dim)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def action_distribution(self, states: torch.Tensor):
        # states: (B, state_dim)
        # actions: (B, action_dim)
        mean, log_std = self.forward(states)  # (B, action_dim) x 2
        dist = torch.distributions.Normal(mean, torch.exp(log_std))
        return dist

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (B, state_dim)
        # actions: (B, action_dim)
        dist = self.action_distribution(states)

        squashed_actions = (actions - self.action_bias) / self.action_scale # (-1, 1)
        squashed_actions = torch.clamp(squashed_actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)

        raw_actions = 0.5 * (torch.log1p(squashed_actions) - torch.log1p(-squashed_actions))

        log_probs = dist.log_prob(raw_actions)
        correction = torch.log(self.action_scale * (1.0 - torch.pow(squashed_actions, 2)))
        log_probs = log_probs - correction
        return log_probs.sum(dim=-1)  # (B,)

    def sample(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        return_log_probs: bool = False,
    ):
        # states: (B, state_dim)
        dist = self.action_distribution(states)

        raw_actions = dist.mean if deterministic else dist.rsample() # (-inf, +inf)
        squashed_actions = torch.tanh(raw_actions)  # (-1; 1)
        actions = self.action_bias + self.action_scale * squashed_actions # (low, high)
        actions = torch.clamp(actions, min=self.action_low, max=self.action_high)

        if not return_log_probs:
            return actions

        log_probs = dist.log_prob(raw_actions)  # (B, action_dim)
        correction = torch.log(self.action_scale * (1.0 - squashed_actions.pow(2)) + 1e-6)  # (B, action_dim)
        log_probs = log_probs - correction
        log_probs = log_probs.sum(dim=-1)  # (B,)

        return actions, log_probs


class OuterOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        fisher_reg: float,
        discount: float,
        alpha: float,
        max_grad_norm=None,
        scheduler_gamma: float = 1.0,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.discount = discount
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.scheduler_gamma = scheduler_gamma

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

        self.optimizer = torch.optim.Adam(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, scheduler_gamma)

        self.outer_step = 0

    def _grad_R_tail_with_discount(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        device = next(self.reward.parameters()).device
        T = states.size(0)

        r_a_s_t = self.reward(states, actions)  # (T,)
        weights = discount_weights(T, self.discount, device)  # (T,)
        r_a_s_t = weights * r_a_s_t

        grad_outputs = torch.eye(T, dtype=r_a_s_t.dtype, device=device)  # (T, T)

        grads = torch.autograd.grad(
            r_a_s_t,
            self.reward.parameters(),
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )

        grads = flat_grad(grads, flat_dim=1).detach()  # (T, reward_dim)
        suffix_sums = torch.flip(torch.cumsum(torch.flip(grads, dims=[0]), dim=0), dims=[0]) # (T, reward_dim)
        return suffix_sums

    def _grad_log_pi_a_s(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        T = states.size(0)

        log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
        grad_outputs = torch.eye(T, dtype=log_pi_a_s.dtype, device=device)  # (T, T)

        grads = torch.autograd.grad(
            log_pi_a_s,
            self.policy.parameters(),
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )

        grads = flat_grad(grads, flat_dim=1).detach()  # (T, policy_dim)
        return grads
    
    def fisher(self, trajs) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        F = torch.zeros(policy_dim, policy_dim, dtype=torch.float64, device=device)

        for traj in tqdm(trajs, desc="Fisher", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions).to(torch.float64)  # (T, policy_dim)
            weights = discount_weights(T, self.discount, device, dtype=torch.float64)  # (T,)
            F += torch.einsum("t,ti,tj->ij", weights, grad_log_pi_a_s, grad_log_pi_a_s)  # (policy_dim, policy_dim)

        F /= len(trajs)
        F = 0.5 * (F + F.T)
        F = self.alpha * F
        F += self.fisher_reg * torch.eye(policy_dim, dtype=torch.float64, device=device)
        F = 0.5 * (F + F.T)
        return F

    def d_outer_d_policy(self, expert_trajs) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        E = torch.zeros(policy_dim, dtype=torch.float32, device=device)

        for traj in tqdm(expert_trajs, desc="Outer grad", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
            weights = discount_weights(T, self.discount, device)  # (T,)
            sum_weighted_log_pi_a_s = (weights * log_pi_a_s).sum()  # (1,)

            grad = torch.autograd.grad(
                sum_weighted_log_pi_a_s,
                self.policy.parameters(),
                retain_graph=False,
                create_graph=False,
            )

            grad = flat_grad(grad).detach()  # (policy_dim,)
            E += grad

        return -(E / len(expert_trajs))

    def d_cross_vec_product(self, trajs, v: torch.Tensor) -> torch.Tensor:
        reward_dim = num_params(self.reward)
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        v = v.to(dtype=torch.float32, device=device)

        if v.numel() != policy_dim:
            raise ValueError(f"`v` must have shape ({policy_dim},), got {tuple(v.shape)}.")

        out = torch.zeros(reward_dim, dtype=torch.float32, device=device)

        for traj in tqdm(trajs, desc="Cross vec product", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)

            grad_R_tail_with_discount = self._grad_R_tail_with_discount(states, actions)  # (T, reward_dim)
            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions)  # (T, policy_dim)
            out += torch.einsum("tr,tp,p->r", grad_R_tail_with_discount, grad_log_pi_a_s, v)  # (reward_dim,)

        return -(out / len(trajs))

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)  # (policy_dim, policy_dim)
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs).to(dtype=torch.float64)  # (policy_dim,)

        fisher_inv_d_outer_d_policy = torch.linalg.solve(fisher, d_outer_d_policy)  # (policy_dim,)
        fisher_inv_d_outer_d_policy = fisher_inv_d_outer_d_policy.to(dtype=torch.float32)

        hypergrad = -self.d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # (reward_dim,)

        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(fisher)
            min_eig = eigvals.min()
            max_eig = eigvals.max()
            cond_number = max_eig / min_eig

            tqdm.write(
                f"Fisher stats | "
                f"min_eig={min_eig.item():.3e} | "
                f"max_eig={max_eig.item():.3e} | "
                f"cond={cond_number.item():.3e} | "
                f"outer_grad_norm={d_outer_d_policy.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e}"
                f"solve_norm={fisher_inv_d_outer_d_policy.norm().item():.3e} | "
                f"solve_abs_max={fisher_inv_d_outer_d_policy.abs().max().item():.3e} | "
            )

            # mlflow.log_metrics(
            #     {
            #         "fisher_min_eig": min_eig.item(),
            #         "fisher_max_eig": max_eig.item(),
            #         "fisher_cond": cond_number.item(),
            #         "outer_grad_norm": d_outer_d_policy.norm().item(),

            #         "fisher_inv_d_outer_d_policy_norm": fisher_inv_d_outer_d_policy.norm().item(),
            #         "fisher_inv_d_outer_d_policy_abs_max": fisher_inv_d_outer_d_policy.abs().max().item(),
            #         "fisher_inv_d_outer_d_policy_mean": fisher_inv_d_outer_d_policy.mean().item(),
            #         "fisher_inv_d_outer_d_policy_std": fisher_inv_d_outer_d_policy.std().item(),

            #         "hypergrad_norm_before_clip": hypergrad.norm().item(),
            #     },
            #     step=self.outer_step
            # )

        return hypergrad

    def step(self, expert_trajs, agent_trajs) -> torch.Tensor:
        hypergradient = self.hypergradient(expert_trajs, agent_trajs)

        self.raw_grad_norm = hypergradient.norm().item()
        if self.max_grad_norm is not None and self.raw_grad_norm > self.max_grad_norm:
            hypergradient = hypergradient * (self.max_grad_norm / self.raw_grad_norm)
        self.clipped_grad_norm = hypergradient.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.reward, hypergradient)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
        
        self.outer_step += 1

        return hypergradient


def train_bilevel(config: dict, logger) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]
    sac_cfg = inner_cfg["sac"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    env_cfg = config["env"]
    ckpt_cfg = config["checkpoint"]

    set_random_seed(int(fisher_cfg["random_seed"]))
    env = Environment(env_cfg["id"], int(fisher_cfg["env_seed"]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    mlflow.log_params({
        "discount": fisher_cfg["discount"],
        "lr_reward": fisher_cfg["lr_reward"],
        "fisher_reg": fisher_cfg["fisher_reg"],
        "n_outer_steps": fisher_cfg["n_outer_steps"],
        "n_inner_steps": fisher_cfg["n_inner_steps"],
        "n_agent_traj": fisher_cfg["n_agent_traj"],
        "reward_hidden": config["reward"]["hidden_dim"],
        "policy_hidden": config["policy"]["hidden_dim"],
        "alpha": fisher_cfg["alpha"],
        "batch_size": sac_cfg["batch_size"],
    })

    if inner_cfg["type"] != "sac":
        raise ValueError(f"Expected fisher.inner.type = sac, got {inner_cfg['type']}")

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

    hidden_dim = int(policy_cfg["hidden_dim"])
    n_layers = int(policy_cfg["n_hidden_layers"])

    reward = Reward(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_hidden_layers=int(reward_cfg["n_hidden_layers"]),
        hidden_dim=int(reward_cfg["hidden_dim"]),
        clamp_magnitude=float(reward_cfg["clamp_magnitude"]),
    ).to(device)

    policy = Policy(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_layers,
        log_std_min=float(policy_cfg["log_std_min"]),
        log_std_max=float(policy_cfg["log_std_max"]),
    ).to(device)

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        discount=float(fisher_cfg["discount"]),
        alpha=float(fisher_cfg["alpha"]),
        max_grad_norm=float(fisher_cfg["max_grad_norm"]),
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
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

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "fisher.pt")
    best_env_reward = float("-inf")

    arch = {
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "policy_hidden": hidden_dim,
        "policy_n_hidden_layers": n_layers,
        "reward_n_hidden_layers": int(reward_cfg["n_hidden_layers"]),
        "reward_hidden": int(reward_cfg["hidden_dim"]),
        "reward_clamp_magnitude": float(reward_cfg["clamp_magnitude"]),
        "log_std_min": float(policy_cfg["log_std_min"]),
        "log_std_max": float(policy_cfg["log_std_max"]),
        "action_low": env.action_low.tolist(),
        "action_high": env.action_high.tolist(),
        "method": "fisher",
        "agent": "sac",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    # def eval_reward_stats():
    #     expert_reward_stats = learned_reward_stats(reward, expert_valid_trajs, outer_optimizer.discount)
    #     random_reward_stats = learned_reward_stats(reward, random_valid_trajs, outer_optimizer.discount)

    #     expert_learned_return = expert_reward_stats["return_mean"]
    #     random_learned_return = random_reward_stats["return_mean"]
    #     expert_random_learned_diff = expert_learned_return - random_learned_return

    #     expert_learned_step_mean = expert_reward_stats["step_mean"]
    #     random_learned_step_mean = random_reward_stats["step_mean"]

    #     logger.info(
    #         f"   [outer] "
    #         f"expert_ret={expert_learned_return:.3f} "
    #         f"random_ret={random_learned_return:.3f} "
    #         f"diff={expert_random_learned_diff:.3f} "
    #         f"expert_step={expert_learned_step_mean:.4f} "
    #         f"random_step={random_learned_step_mean:.4f}"
    #     )

    def log_and_checkpoint(outer_step: int, agent_trajs):
        nonlocal best_env_reward

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss(policy, expert_train_trajs, outer_optimizer.discount)

        agent_len = mean_trajectory_length(agent_trajs)
        expert_len = mean_trajectory_length(expert_train_trajs)

        agent_ret = mean_trajectory_return(agent_trajs)
        expert_ret = mean_trajectory_return(expert_train_trajs)

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

        logger.info(row)

        mlflow.log_metrics(
            {
                "outer_loss": l_outer,
                "agent_return": agent_ret,
                "expert_return": expert_ret,
                "agent_length": agent_len,
                "rank_corr": rank_corr_val,
                "policy_nll": policy_nll_val,
                "hypergrad_norm": raw_hypgrad_norm,
                "hypergrad_norm_clipped": clipped_hypgrad_norm,
                "lr_reward": lr_outer_current,
            },
            step=outer_step,
        )

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )

    logger.info(header)

    # eval_reward_stats()

    def inner_optimize(outer_step: int):
        sac = SAC(
            policy=policy,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=sac_cfg["q_hidden_dim"],
            n_hidden_layers=sac_cfg["q_n_hidden_layers"],
            gamma=fisher_cfg["discount"],
            alpha=fisher_cfg["alpha"],
            tau=sac_cfg["tau"],
            replay_buffer_capacity=sac_cfg["replay_buffer_capacity"],
        )

        sac_train_env = Environment(id=config["env"]["id"], seed=int(inner_cfg["eval_env_seed"]), custom_reward_fn=reward.as_fn())
        sac_eval_env = Environment(id=config["env"]["id"], seed=int(inner_cfg["train_env_seed"]), custom_reward_fn=None)

        @torch.no_grad()
        def validate(ts: int, n_eval_traj: int = 10):
            sac.policy.eval()

            agent_valid_trajs = collect_trajectories(
                env=sac_eval_env,
                policy=sac.policy,
                n=n_eval_traj,
                max_steps=config["env"]["max_steps"],
                verbose=False,
            )

            l_inner = inner_loss(
                policy=sac.policy,
                reward=reward,
                trajs=agent_valid_trajs,
                discount=sac.gamma,
                alpha=sac.alpha,
            )

            l_outer = outer_loss(
                policy=sac.policy,
                expert_trajs=expert_valid_trajs,
                discount=sac.gamma,
            )

            mlflow.log_metrics(
                {
                    f"sac_{outer_step}/l_inner": float(l_inner),
                    f"sac_{outer_step}/l_outer": float(l_outer),
                },
                step=ts,
            )

            sac.policy.train()

        sac.optimize(
            train_env=sac_train_env,
            total_steps=n_inner_steps,
            learning_starts=int(sac_cfg["learning_starts"]),
            batch_size=int(sac_cfg["batch_size"]),
            max_grad_norm=sac_cfg["max_grad_norm"],
            gradient_update_steps=int(sac_cfg["gradient_update_steps"]),
            target_update_interval=int(sac_cfg["target_update_interval"]),
            critic_lr=float(sac_cfg["critic_lr"]),
            actor_lr=float(sac_cfg["actor_lr"]),
            validate_fn=validate,
            validate_every=1000
        )

        sac_train_env.close()
        sac_eval_env.close()

    inner_optimize(outer_step=0)

    agent_trajs = collect_trajectories(
        env=env,
        policy=policy,
        n=n_agent_traj,
        max_steps=int(config["env"]["max_steps"]),
        desc="agent outer trajs",
    )

    log_and_checkpoint(outer_step=0, agent_trajs=agent_trajs)

    for outer_step in range(1, n_outer_steps + 1):
        outer_optimizer.step(expert_train_trajs, agent_trajs)
        # eval_reward_stats()

        inner_optimize(outer_step=outer_step)

        agent_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_traj,
            max_steps=int(config["env"]["max_steps"]),
            desc="agent outer trajs",
        )

        log_and_checkpoint(outer_step=outer_step, agent_trajs=agent_trajs)

    env.close()
    
    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD IRL SAC — Hopper")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("hopper", args.config)
    config = load_config(config_path)

    log_cfg = config["logging"]

    log_dir = log_cfg["log_dir"]
    logger = get_logger("fisher_hopper", log_dir=log_dir)

    logger.info("=== Fisher-NHD Hopper SAC ===")

    mlflow.set_experiment("fisher")
    with mlflow.start_run(run_name="hopper"):
        history = train_bilevel(config, logger)

    report_path = Path(log_cfg["report_dir"]) / "fisher_sac_hopper_history.json"
    save_history(history, str(report_path))
    logger.info(f"History saved to {report_path}")


if __name__ == "__main__":
    main()
