"""
Fisher-NHD IRL with SAC inner agent for Hopper.

Usage:
    python -m src.irl.hopper.fisher
    python -m src.irl.hopper.fisher --config configs/hopper.yaml
"""

import argparse
import copy
import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
import mlflow

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import (
    inner_loss,
    learned_reward_observation_stats,
    learned_reward_stats,
    observation_outer_loss,
    outer_loss,
    policy_entropy,
    policy_nll,
    rank_corr,
)
from src.agents.sac import SAC, observation_normalization_kwargs
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import Environment
from src.utils.logging import get_logger, save_history
from src.utils.mlflow import begin_mlflow_run, log_artifact_if_exists
from src.utils.sb3 import init_policy_from_sb3_sac_expert
from src.utils.seeding import set_random_seed
from src.utils.torch import flat_grad, num_params, assign_flat_gradients, to_device
from src.utils.trajectories import (
    collect_trajectories,
    mean_trajectory_length,
    mean_trajectory_return,
    discount_weights,
)
from src.utils.visualization import log_policy_trajectory_gifs


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
            device = next(self.parameters()).device
            was_training = self.training
            if was_training:
                self.eval()

            with torch.no_grad():
                states = states.to(device)
                actions = actions.to(device)
                out = self.rewards(states, actions)

            if was_training:
                self.train()

            return out.detach().cpu() # (B,)
        
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
        activation: str = "relu",
        normalize_observations: bool = False,
        observation_norm_epsilon: float = 1e-8,
        observation_norm_clip: float | None = 10.0,
    ):
        super().__init__()

        activation = activation.lower()
        if activation == "relu":
            activation_cls = nn.ReLU
        elif activation == "tanh":
            activation_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported policy activation: {activation}")

        in_dim = state_dim
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), activation_cls()]
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
        self.activation = activation
        self.state_dim = int(state_dim)
        self.observation_normalization_enabled = bool(normalize_observations)
        self.observation_norm_epsilon = float(observation_norm_epsilon)
        self.observation_norm_clip = observation_norm_clip

        if self.observation_norm_epsilon <= 0.0:
            raise ValueError("observation_norm_epsilon must be positive.")
        if self.observation_norm_clip is not None and self.observation_norm_clip <= 0.0:
            raise ValueError("observation_norm_clip must be positive or None.")

        if self.observation_normalization_enabled:
            self.register_buffer(
                "observation_mean",
                torch.zeros(self.state_dim, dtype=torch.float64),
            )
            self.register_buffer(
                "observation_var",
                torch.ones(self.state_dim, dtype=torch.float64),
            )
            self.register_buffer(
                "observation_count",
                torch.tensor(1e-4, dtype=torch.float64),
            )

    @torch.no_grad()
    def update_observation_stats(self, observations: torch.Tensor) -> None:
        if not self.observation_normalization_enabled:
            return

        values = torch.as_tensor(
            observations,
            dtype=torch.float64,
            device=self.observation_mean.device,
        ).reshape(-1, self.state_dim)
        if values.shape[0] == 0:
            return

        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = float(values.shape[0])
        delta = batch_mean - self.observation_mean
        total_count = self.observation_count + batch_count

        new_mean = self.observation_mean + delta * batch_count / total_count
        current_m2 = self.observation_var * self.observation_count
        batch_m2 = batch_var * batch_count
        correction = delta.square() * self.observation_count * batch_count / total_count
        new_var = (current_m2 + batch_m2 + correction) / total_count

        self.observation_mean.copy_(new_mean)
        self.observation_var.copy_(new_var.clamp_min(0.0))
        self.observation_count.copy_(total_count)

    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.observation_normalization_enabled:
            return observations

        mean = self.observation_mean.to(
            device=observations.device,
            dtype=observations.dtype,
        )
        variance = self.observation_var.to(
            device=observations.device,
            dtype=observations.dtype,
        )
        normalized = (observations - mean) / torch.sqrt(
            variance + self.observation_norm_epsilon
        )
        if self.observation_norm_clip is not None:
            normalized = torch.clamp(
                normalized,
                -self.observation_norm_clip,
                self.observation_norm_clip,
            )
        return normalized

    def forward(self, states: torch.Tensor):
        x = self.backbone(self.normalize_observations(states))
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
        n_cg_steps: int,
        cg_tol: float,
        max_cg_relative_residual: float = float("inf"),
        max_grad_norm=None,
        scheduler_gamma: float = 1.0,
        normalize_discounted_sums: bool = False,
        estimator_weighting: str = "discounted_trajectories",
        gradient_method: str = "fisher",
        weight_decay: float = 0.0,
        enable_mlflow: bool = True,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.discount = discount
        self.alpha = alpha
        self.n_cg_steps = n_cg_steps
        self.cg_tol = cg_tol
        self.max_cg_relative_residual = max_cg_relative_residual
        self.max_grad_norm = max_grad_norm
        self.scheduler_gamma = scheduler_gamma
        self.normalize_discounted_sums = normalize_discounted_sums
        self.estimator_weighting = str(estimator_weighting).lower().replace("-", "_")
        self.gradient_method = str(gradient_method).lower().replace("-", "_")
        self.weight_decay = float(weight_decay)
        self.enable_mlflow = enable_mlflow

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0
        self.parameter_update_norm = 0.0
        self.ml_counterfactual_update_norm = float("nan")
        self.ml_irl_update_direction_norm = float("nan")
        self.last_ml_irl_update_direction = None
        self.fisher_ml_cosine = float("nan")
        self.cg_relative_residual = float("nan")
        self.cg_update_skipped = False

        if self.gradient_method not in {"fisher", "ml_irl"}:
            raise ValueError(
                "gradient_method must be either 'fisher' or 'ml_irl', "
                f"got {gradient_method!r}."
            )
        if self.estimator_weighting not in {
            "discounted_trajectories",
            "uniform_observations",
        }:
            raise ValueError(
                "estimator_weighting must be either 'discounted_trajectories' "
                "or 'uniform_observations', "
                f"got {estimator_weighting!r}."
            )
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be nonnegative, got {weight_decay}.")
        if self.max_cg_relative_residual <= 0.0:
            raise ValueError(
                "max_cg_relative_residual must be positive, "
                f"got {self.max_cg_relative_residual}."
            )

        self.optimizer = torch.optim.Adam(
            self.reward.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, scheduler_gamma)

        self.outer_step = 0

    @staticmethod
    def _stat_value(value) -> float:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 0:
                return float("nan")
            return float(value.item())
        return float(value)

    def _tensor_stats(self, name: str, tensor: torch.Tensor) -> dict[str, float]:
        tensor = tensor.detach()
        norm = tensor.float().norm() if tensor.numel() else torch.tensor(float("nan"), device=tensor.device)
        return {f"{name}/norm": self._stat_value(norm)}

    def _log_stats(self, stats: dict[str, float]) -> None:
        tqdm.write(
            " | ".join(
                f"{key}={value:.3e}" for key, value in sorted(stats.items())
            )
        )

        finite_stats = {
            key: value for key, value in stats.items()
            if np.isfinite(value)
        }

        if finite_stats and self.enable_mlflow:
            mlflow.log_metrics(finite_stats, step=self.outer_step)

    def _discount_normalizer(self, weights: torch.Tensor) -> torch.Tensor:
        if not self.normalize_discounted_sums:
            return torch.ones((), dtype=weights.dtype, device=weights.device)

        return weights.sum().clamp_min(torch.finfo(weights.dtype).eps)

    def _trajectory_weights(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.estimator_weighting == "uniform_observations":
            return torch.ones(length, dtype=dtype, device=device)

        weights = discount_weights(
            length,
            self.discount,
            device,
            dtype=dtype,
        )
        return weights / self._discount_normalizer(weights)

    def _estimator_normalizer(self, trajs) -> int:
        if not trajs:
            raise ValueError("At least one trajectory is required.")

        if self.estimator_weighting == "uniform_observations":
            observation_count = sum(len(traj["states"]) for traj in trajs)
            if observation_count == 0:
                raise ValueError("Trajectories must contain at least one observation.")
            return observation_count

        return len(trajs)

    def _precomputed_score_normalizer(
        self,
        precomputed_scores: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        if not precomputed_scores:
            raise ValueError("At least one precomputed Fisher trajectory is required.")

        if self.estimator_weighting == "uniform_observations":
            observation_count = sum(scores.size(0) for scores, _ in precomputed_scores)
            if observation_count == 0:
                raise ValueError("Fisher trajectories must contain at least one observation.")
            return observation_count

        return len(precomputed_scores)

    def _clip_flat_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        gradient_norm = gradient.norm().item()
        if self.max_grad_norm is not None and gradient_norm > self.max_grad_norm:
            return gradient * (self.max_grad_norm / gradient_norm)
        return gradient

    def _counterfactual_adam_update_norm(self, gradient: torch.Tensor) -> float:
        shadow_reward = copy.deepcopy(self.reward)
        shadow_optimizer = torch.optim.Adam(shadow_reward.parameters())
        shadow_optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        shadow_optimizer.zero_grad()

        gradient = self._clip_flat_gradient(gradient)
        assign_flat_gradients(shadow_reward, gradient)

        with torch.no_grad():
            parameters_before = torch.cat(
                [parameter.detach().reshape(-1).clone() for parameter in shadow_reward.parameters()]
            )

        shadow_optimizer.step()

        with torch.no_grad():
            parameters_after = torch.cat(
                [parameter.detach().reshape(-1) for parameter in shadow_reward.parameters()]
            )
            return (parameters_after - parameters_before).norm().item()

    def _mean_reward_gradient(self, trajs, desc: str) -> torch.Tensor:
        if not trajs:
            raise ValueError("At least one trajectory is required.")

        reward_dim = num_params(self.reward)
        device = next(self.reward.parameters()).device
        mean_gradient = torch.zeros(reward_dim, dtype=torch.float32, device=device)

        for traj in tqdm(trajs, desc=desc, leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            rewards = self.reward(states, actions)
            weights = self._trajectory_weights(
                rewards.size(0),
                device,
                rewards.dtype,
            )
            weighted_reward = (weights * rewards).sum()

            gradient = torch.autograd.grad(
                weighted_reward,
                self.reward.parameters(),
                retain_graph=False,
                create_graph=False,
            )
            mean_gradient += flat_grad(gradient).detach()

        return mean_gradient / self._estimator_normalizer(trajs)

    def ml_irl_update_direction(self, expert_trajs, agent_trajs) -> torch.Tensor:
        expert_gradient = self._mean_reward_gradient(
            expert_trajs,
            desc="ML-IRL expert gradient",
        )
        agent_gradient = self._mean_reward_gradient(
            agent_trajs,
            desc="ML-IRL agent gradient",
        )
        return expert_gradient - agent_gradient

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

        grad_log_norms = []
        fisher_terms = []

        for traj in tqdm(trajs, desc="Fisher", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions).to(torch.float64)  # (T, policy_dim)
            weights = self._trajectory_weights(T, device, torch.float64)  # (T,)
            fisher_term = torch.einsum("t,ti,tj->ij", weights, grad_log_pi_a_s, grad_log_pi_a_s)  # (policy_dim, policy_dim)
            F += fisher_term

            with torch.no_grad():
                grad_log_norms.append(grad_log_pi_a_s.float().norm())
                fisher_terms.append(fisher_term.float().norm())

        F /= self._estimator_normalizer(trajs)
        F = 0.5 * (F + F.T)
        F = self.alpha * F
        F += self.fisher_reg * torch.eye(policy_dim, dtype=torch.float64, device=device)
        F = 0.5 * (F + F.T)

        with torch.no_grad():
            stats = {}
            stats.update(self._tensor_stats("fisher/grad_log_traj_norms", torch.stack(grad_log_norms)))
            stats.update(self._tensor_stats("fisher/traj_term_norms", torch.stack(fisher_terms)))
            stats.update(self._tensor_stats("fisher/matrix", F))
            self._log_stats(stats)

        return F

    def _precompute_fisher_scores(
        self,
        trajs,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not trajs:
            raise ValueError("At least one Fisher trajectory is required.")

        device = next(self.policy.parameters()).device
        precomputed_scores = []

        for traj in tqdm(trajs, desc="Precompute Fisher scores", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions).to(torch.float64)
            weights = self._trajectory_weights(T, device, torch.float64)
            precomputed_scores.append((grad_log_pi_a_s, weights))

        n_timesteps = sum(scores.size(0) for scores, _ in precomputed_scores)
        storage_bytes = sum(
            tensor.numel() * tensor.element_size()
            for scores, weights in precomputed_scores
            for tensor in (scores, weights)
        )
        tqdm.write(
            f"Precomputed Fisher scores | "
            f"trajs={len(precomputed_scores)} | "
            f"timesteps={n_timesteps} | "
            f"storage={storage_bytes / 1024**3:.2f} GiB"
        )

        return precomputed_scores

    def _fisher_vector_product_from_scores(
        self,
        precomputed_scores: list[tuple[torch.Tensor, torch.Tensor]],
        vector: torch.Tensor,
    ) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        vector = vector.to(dtype=torch.float64, device=device)

        if vector.numel() != policy_dim:
            raise ValueError(f"`vector` must have shape ({policy_dim},), got {tuple(vector.shape)}.")

        out = torch.zeros_like(vector)

        for grad_log_pi_a_s, weights in precomputed_scores:
            score_vector_product = torch.einsum("tp,p->t", grad_log_pi_a_s, vector)  # (T,)
            out += torch.einsum("t,tp,t->p", weights, grad_log_pi_a_s, score_vector_product)

        out /= self._precomputed_score_normalizer(precomputed_scores)
        out = self.alpha * out
        return out + self.fisher_reg * vector

    def fisher_vector_product(self, trajs, vector: torch.Tensor) -> torch.Tensor:
        precomputed_scores = self._precompute_fisher_scores(trajs)
        return self._fisher_vector_product_from_scores(precomputed_scores, vector)

    def conjugate_gradient_solve(self, trajs, rhs: torch.Tensor) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        solution = torch.zeros_like(rhs, dtype=torch.float64, device=device)
        residual = rhs.to(dtype=torch.float64, device=device).clone()
        direction = residual.clone()
        residual_dot = torch.dot(residual, residual)
        initial_residual_norm = residual_dot.sqrt()
        target_residual_norm = self.cg_tol * initial_residual_norm.clamp_min(1.0)

        cg_iterations = 0
        last_curvature = torch.tensor(float("nan"), dtype=torch.float64, device=device)

        if initial_residual_norm.item() <= target_residual_norm.item():
            self.cg_relative_residual = 0.0
            return solution

        precomputed_scores = self._precompute_fisher_scores(trajs)

        for cg_iteration in range(self.n_cg_steps):
            fisher_direction = self._fisher_vector_product_from_scores(
                precomputed_scores,
                direction,
            )
            curvature = torch.dot(direction, fisher_direction)
            last_curvature = curvature.detach()

            if not torch.isfinite(curvature).item() or curvature.item() <= 0.0:
                break

            step_size = residual_dot / curvature
            solution = solution + step_size * direction
            residual = residual - step_size * fisher_direction
            new_residual_dot = torch.dot(residual, residual)
            cg_iterations = cg_iteration + 1

            if new_residual_dot.sqrt().item() <= target_residual_norm.item():
                residual_dot = new_residual_dot
                break

            direction = residual + (new_residual_dot / residual_dot) * direction
            residual_dot = new_residual_dot

        with torch.no_grad():
            final_residual_norm = residual_dot.sqrt()
            relative_residual = final_residual_norm / initial_residual_norm.clamp_min(torch.finfo(torch.float64).eps)
            self.cg_relative_residual = relative_residual.item()
            stats = {
                "solve/cg_iterations": float(cg_iterations),
                "solve/cg_residual_norm": final_residual_norm.item(),
                "solve/cg_relative_residual": relative_residual.item(),
                "solve/cg_last_curvature": last_curvature.item(),
            }
            self._log_stats(stats)

            tqdm.write(
                f"CG solve | "
                f"iters={cg_iterations} | "
                f"residual={final_residual_norm.item():.3e} | "
                f"relative_residual={relative_residual.item():.3e} | "
                f"outer_grad_norm={rhs.norm().item():.3e}"
            )

        return solution

    def d_outer_d_policy(self, expert_trajs) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        E = torch.zeros(policy_dim, dtype=torch.float32, device=device)
        traj_grad_norms = []
        traj_logprob_sums = []

        for traj in tqdm(expert_trajs, desc="Outer grad", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
            weights = self._trajectory_weights(T, device, log_pi_a_s.dtype)  # (T,)
            sum_weighted_log_pi_a_s = (weights * log_pi_a_s).sum()  # (1,)

            grad = torch.autograd.grad(
                sum_weighted_log_pi_a_s,
                self.policy.parameters(),
                retain_graph=False,
                create_graph=False,
            )

            grad = flat_grad(grad).detach()  # (policy_dim,)
            E += grad

            with torch.no_grad():
                traj_grad_norms.append(grad.norm())
                traj_logprob_sums.append(sum_weighted_log_pi_a_s.detach())

        out = -(E / self._estimator_normalizer(expert_trajs))

        with torch.no_grad():
            stats = {}
            stats.update(self._tensor_stats("outer/traj_grad_norms", torch.stack(traj_grad_norms)))
            stats.update(self._tensor_stats("outer/traj_weighted_logprob_sums", torch.stack(traj_logprob_sums)))
            stats.update(self._tensor_stats("outer/d_outer_d_policy", out))
            self._log_stats(stats)

        return out

    def d_cross_vec_product(self, trajs, v: torch.Tensor) -> torch.Tensor:
        reward_dim = num_params(self.reward)
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        v = v.to(dtype=torch.float32, device=device)

        if v.numel() != policy_dim:
            raise ValueError(f"`v` must have shape ({policy_dim},), got {tuple(v.shape)}.")

        out = torch.zeros(reward_dim, dtype=torch.float32, device=device)
        grad_log_norms = []
        score_v_norms = []
        scalar_objectives = []
        traj_cross_norms = []

        for traj in tqdm(trajs, desc="Cross vec product", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions)  # (T, policy_dim)
            score_v = torch.einsum("tp,p->t", grad_log_pi_a_s, v)  # (T,)
            score_v_prefix = torch.cumsum(score_v, dim=0)  # (T,)

            rewards = self.reward(states, actions)  # (T,)
            weights = self._trajectory_weights(T, device, rewards.dtype)
            scalar_objective = torch.sum(weights * rewards * score_v_prefix)

            traj_cross = flat_grad(
                torch.autograd.grad(
                    scalar_objective,
                    self.reward.parameters(),
                    retain_graph=False,
                    create_graph=False,
                )
            ).detach()
            out += traj_cross

            with torch.no_grad():
                grad_log_norms.append(grad_log_pi_a_s.norm())
                score_v_norms.append(score_v.norm())
                scalar_objectives.append(scalar_objective.detach())
                traj_cross_norms.append(traj_cross.norm())

        out = -(out / self._estimator_normalizer(trajs))

        with torch.no_grad():
            stats = {}
            stats.update(self._tensor_stats("cross/grad_log_traj_norms", torch.stack(grad_log_norms)))
            stats.update(self._tensor_stats("cross/score_v_traj_norms", torch.stack(score_v_norms)))
            stats.update(self._tensor_stats("cross/scalar_objectives", torch.stack(scalar_objectives)))
            stats.update(self._tensor_stats("cross/traj_cross_norms", torch.stack(traj_cross_norms)))
            stats.update(self._tensor_stats("cross/d_cross_vec_product", out))
            self._log_stats(stats)

        return out

    def hypergradient(self, expert_trajs, agent_trajs, fisher_trajs) -> torch.Tensor | None:
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs).to(dtype=torch.float64)  # (policy_dim,)

        fisher_inv_d_outer_d_policy = self.conjugate_gradient_solve(fisher_trajs, d_outer_d_policy)  # (policy_dim,)
        fisher_inv_d_outer_d_policy = fisher_inv_d_outer_d_policy.to(dtype=torch.float32)

        self.cg_update_skipped = (
            not np.isfinite(self.cg_relative_residual)
            or self.cg_relative_residual > self.max_cg_relative_residual
        )
        if self.cg_update_skipped:
            ml_irl_update_direction = self.ml_irl_update_direction(expert_trajs, agent_trajs)
            self.ml_irl_update_direction_norm = ml_irl_update_direction.norm().item()
            self.last_ml_irl_update_direction = ml_irl_update_direction.detach()
            self.fisher_ml_cosine = float("nan")
            self._log_stats(
                {
                    "gradient/ml_irl_update_direction_norm": self.ml_irl_update_direction_norm,
                    "solve/cg_update_skipped": 1.0,
                }
            )
            tqdm.write(
                f"Skipping Fisher cross product and reward update | "
                f"cg_relative_residual={self.cg_relative_residual:.3e} | "
                f"maximum={self.max_cg_relative_residual:.3e}"
            )
            return None

        hypergrad = -self.d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # (reward_dim,)
        fisher_update_direction = -hypergrad
        ml_irl_update_direction = self.ml_irl_update_direction(expert_trajs, agent_trajs)

        with torch.no_grad():
            fisher_update_norm = fisher_update_direction.norm()
            ml_irl_update_norm = ml_irl_update_direction.norm()
            alignment_denominator = fisher_update_norm * ml_irl_update_norm
            if alignment_denominator.item() > 0.0:
                fisher_ml_cosine = torch.dot(
                    fisher_update_direction,
                    ml_irl_update_direction,
                ) / alignment_denominator
            else:
                fisher_ml_cosine = torch.tensor(float("nan"), device=hypergrad.device)

            self.ml_irl_update_direction_norm = ml_irl_update_norm.item()
            self.last_ml_irl_update_direction = ml_irl_update_direction.detach()
            self.fisher_ml_cosine = fisher_ml_cosine.item()

            stats = {}
            stats.update(self._tensor_stats("solve/fisher_inv_d_outer_d_policy", fisher_inv_d_outer_d_policy))
            stats.update(self._tensor_stats("hypergrad/raw", hypergrad))
            stats.update(
                {
                    "gradient/fisher_update_direction_norm": fisher_update_norm.item(),
                    "gradient/ml_irl_update_direction_norm": ml_irl_update_norm.item(),
                    "gradient/fisher_ml_cosine": fisher_ml_cosine.item(),
                }
            )
            self._log_stats(stats)

            tqdm.write(
                f"Fisher CG stats | "
                f"outer_grad_norm={d_outer_d_policy.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e} | "
                f"solve_norm={fisher_inv_d_outer_d_policy.norm().item():.3e}"
            )
            tqdm.write(
                f"Gradient alignment | "
                f"fisher_update_norm={fisher_update_norm.item():.3e} | "
                f"ml_irl_update_norm={ml_irl_update_norm.item():.3e} | "
                f"cosine={fisher_ml_cosine.item():.6f}"
            )

        return hypergrad

    def ml_irl_gradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        ml_irl_update_direction = self.ml_irl_update_direction(
            expert_trajs,
            agent_trajs,
        )
        gradient = -ml_irl_update_direction

        self.ml_irl_update_direction_norm = ml_irl_update_direction.norm().item()
        self.last_ml_irl_update_direction = ml_irl_update_direction.detach()
        self.fisher_ml_cosine = float("nan")
        self.cg_relative_residual = float("nan")
        self.cg_update_skipped = False
        self._log_stats(
            {
                "gradient/ml_irl_update_direction_norm": self.ml_irl_update_direction_norm,
                "gradient/selected_gradient_norm": gradient.norm().item(),
                "gradient/expert_observation_count": float(
                    sum(len(traj["states"]) for traj in expert_trajs)
                ),
                "gradient/agent_observation_count": float(
                    sum(len(traj["states"]) for traj in agent_trajs)
                ),
            }
        )
        tqdm.write(
            "ML-IRL reward gradient | "
            f"update_direction_norm={self.ml_irl_update_direction_norm:.3e}"
        )
        return gradient

    def step(self, expert_trajs, agent_trajs, fisher_trajs) -> torch.Tensor | None:
        if self.gradient_method == "fisher":
            gradient = self.hypergradient(expert_trajs, agent_trajs, fisher_trajs)
        else:
            gradient = self.ml_irl_gradient(expert_trajs, agent_trajs)

        if gradient is None:
            self.raw_grad_norm = float("nan")
            self.clipped_grad_norm = 0.0
            self.parameter_update_norm = 0.0
            self.ml_counterfactual_update_norm = self._counterfactual_adam_update_norm(
                -self.last_ml_irl_update_direction
            )
            self._log_stats(
                {
                    "reward/parameter_update_norm": self.parameter_update_norm,
                    "reward/fisher_parameter_update_norm": self.parameter_update_norm,
                    "reward/ml_counterfactual_update_norm": self.ml_counterfactual_update_norm,
                }
            )
            self.outer_step += 1
            return None

        if self.gradient_method == "fisher":
            self._log_stats({"solve/cg_update_skipped": 0.0})

        self.raw_grad_norm = gradient.norm().item()
        gradient = self._clip_flat_gradient(gradient)
        self.clipped_grad_norm = gradient.norm().item()

        self.optimizer.zero_grad()
        self.ml_counterfactual_update_norm = self._counterfactual_adam_update_norm(
            -self.last_ml_irl_update_direction
        )
        assign_flat_gradients(self.reward, gradient)

        with torch.no_grad():
            parameters_before = torch.cat(
                [parameter.detach().reshape(-1).clone() for parameter in self.reward.parameters()]
            )

        self.optimizer.step()

        with torch.no_grad():
            parameters_after = torch.cat(
                [parameter.detach().reshape(-1) for parameter in self.reward.parameters()]
            )
            self.parameter_update_norm = (parameters_after - parameters_before).norm().item()
            update_stats = {
                "reward/parameter_update_norm": self.parameter_update_norm,
                "reward/ml_counterfactual_update_norm": self.ml_counterfactual_update_norm,
            }
            if self.gradient_method == "fisher":
                update_stats["reward/fisher_parameter_update_norm"] = self.parameter_update_norm
            self._log_stats(update_stats)

        if self.scheduler:
            self.scheduler.step()
        
        self.outer_step += 1

        return gradient


def train_bilevel(
    config: dict,
    logger,
    *,
    enable_mlflow: bool = True,
    metrics_callback: Callable[[int, dict], None] | None = None,
    save_best_checkpoint: bool = False,
) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]
    sac_cfg = inner_cfg["sac"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    env_cfg = config["env"]
    ckpt_cfg = config["checkpoint"]
    log_cfg = config["logging"]
    gradient_method = str(fisher_cfg.get("gradient_method", "fisher")).lower().replace("-", "_")
    if gradient_method not in {"fisher", "ml_irl"}:
        raise ValueError(
            "fisher.gradient_method must be either 'fisher' or 'ml_irl', "
            f"got {fisher_cfg.get('gradient_method')!r}."
        )
    estimator_weighting = str(
        fisher_cfg.get("estimator_weighting", "discounted_trajectories")
    ).lower().replace("-", "_")
    if estimator_weighting not in {
        "discounted_trajectories",
        "uniform_observations",
    }:
        raise ValueError(
            "fisher.estimator_weighting must be either 'discounted_trajectories' "
            "or 'uniform_observations', "
            f"got {fisher_cfg.get('estimator_weighting')!r}."
        )
    run_label = (
        gradient_method
        if estimator_weighting == "discounted_trajectories"
        else f"{gradient_method}_{estimator_weighting}"
    )

    set_random_seed(int(fisher_cfg["random_seed"]))
    env = Environment(env_cfg["id"], int(fisher_cfg["env_seed"]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if enable_mlflow:
        mlflow.log_params({
            "discount": fisher_cfg["discount"],
            "lr_reward": fisher_cfg["lr_reward"],
            "fisher_reg": fisher_cfg["fisher_reg"],
            "n_cg_steps": fisher_cfg.get("n_cg_steps", 10),
            "cg_tol": fisher_cfg.get("cg_tol", 1e-8),
            "max_cg_relative_residual": fisher_cfg.get("max_cg_relative_residual", 1e-2),
            "normalize_discounted_sums": fisher_cfg.get("normalize_discounted_sums", False),
            "estimator_weighting": estimator_weighting,
            "gradient_method": gradient_method,
            "reward_weight_decay": fisher_cfg.get("reward_weight_decay", 0.0),
            "n_outer_steps": fisher_cfg["n_outer_steps"],
            "n_inner_steps": fisher_cfg["n_inner_steps"],
            "n_agent_traj": fisher_cfg["n_agent_traj"],
            "n_fisher_traj": fisher_cfg.get("n_fisher_traj", fisher_cfg["n_agent_traj"]),
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
    n_fisher_traj = int(fisher_cfg.get("n_fisher_traj", n_agent_traj))
    visualization_traj_frequency = int(
        fisher_cfg.get("visualization_traj_frequency", 1)
    )

    if visualization_traj_frequency <= 0:
        raise ValueError(
            "fisher.visualization_traj_frequency must be greater than zero."
        )

    if n_agent_traj <= 0:
        raise ValueError(f"n_agent_traj must be positive, got {n_agent_traj}.")
    if n_fisher_traj <= 0:
        raise ValueError(f"n_fisher_traj must be positive, got {n_fisher_traj}.")

    hidden_dim = int(policy_cfg["hidden_dim"])
    n_layers = int(policy_cfg["n_hidden_layers"])
    observation_norm_kwargs = observation_normalization_kwargs(sac_cfg)

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
        activation=str(policy_cfg.get("activation", "relu")),
        **observation_norm_kwargs,
    ).to(device)

    init_from_expert = bool(fisher_cfg.get("init_policy_from_expert", False))
    if init_from_expert:
        expert_cfg = config["expert"]
        expert_algo = str(expert_cfg["algo"]).lower()
        if expert_algo != "sac":
            raise ValueError(f"Can only initialize Hopper SAC policy from SAC expert, got expert.algo={expert_algo}.")

        init_policy_from_sb3_sac_expert(policy, expert_cfg["save_path"])
        logger.info(f"Initialized policy from SB3 SAC expert: {expert_cfg['save_path']}")
        if enable_mlflow:
            mlflow.log_metric("policy/init_from_expert", 1.0, step=0)
    else:
        if enable_mlflow:
            mlflow.log_metric("policy/init_from_expert", 0.0, step=0)

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        discount=float(fisher_cfg["discount"]),
        alpha=float(fisher_cfg["alpha"]),
        n_cg_steps=int(fisher_cfg.get("n_cg_steps", 10)),
        cg_tol=float(fisher_cfg.get("cg_tol", 1e-8)),
        max_cg_relative_residual=float(fisher_cfg.get("max_cg_relative_residual", 1e-2)),
        max_grad_norm=float(fisher_cfg["max_grad_norm"]),
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
        normalize_discounted_sums=bool(fisher_cfg.get("normalize_discounted_sums", False)),
        estimator_weighting=estimator_weighting,
        gradient_method=gradient_method,
        weight_decay=float(fisher_cfg.get("reward_weight_decay", 0.0)),
        enable_mlflow=enable_mlflow,
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
        "reward/parameter_update_norm": [],
        "reward/fisher_parameter_update_norm": [],
        "reward/ml_counterfactual_update_norm": [],
        "policy_entropy": [],
        "ml_irl_update_direction_norm": [],
        "fisher_ml_cosine": [],
        "lr_outer": [],
        "timing/inner_loop_seconds": [],
        "timing/reward_update_seconds": [],
        "timing/trajectory_collection_seconds": [],
        "timing/visualization_seconds": [],
        "timing/evaluation_checkpoint_seconds": [],
        "timing/outer_step_seconds": [],
    }

    def profiling_time() -> float:
        # CUDA kernels are asynchronous, so synchronize at phase boundaries to
        # make the reported wall-clock durations meaningful.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return perf_counter()

    def record_timing(
        outer_step: int,
        *,
        inner_loop_seconds: float,
        reward_update_seconds: float,
        trajectory_collection_seconds: float,
        visualization_seconds: float,
        evaluation_checkpoint_seconds: float,
        outer_step_seconds: float,
    ) -> None:
        metrics = {
            "timing/inner_loop_seconds": inner_loop_seconds,
            "timing/reward_update_seconds": reward_update_seconds,
            "timing/trajectory_collection_seconds": trajectory_collection_seconds,
            "timing/visualization_seconds": visualization_seconds,
            "timing/evaluation_checkpoint_seconds": evaluation_checkpoint_seconds,
            "timing/outer_step_seconds": outer_step_seconds,
        }
        for name, value in metrics.items():
            history[name].append(value)

        if enable_mlflow:
            mlflow.log_metrics(metrics, step=outer_step)
        logger.info(
            "Timing | "
            f"outer_step={outer_step} | "
            f"inner={inner_loop_seconds:.3f}s | "
            f"reward_update={reward_update_seconds:.3f}s | "
            f"collection={trajectory_collection_seconds:.3f}s | "
            f"visualization={visualization_seconds:.3f}s | "
            f"eval_checkpoint={evaluation_checkpoint_seconds:.3f}s | "
            f"outer_total={outer_step_seconds:.3f}s"
        )

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    active_mlflow_run = mlflow.active_run() if enable_mlflow else None
    if slurm_job_id:
        checkpoint_run_id = f"job_{slurm_job_id}"
    elif active_mlflow_run is not None:
        checkpoint_run_id = f"run_{active_mlflow_run.info.run_id}"
    else:
        checkpoint_run_id = "local"

    checkpoint_prefix = f"{run_label}_sac_{checkpoint_run_id}"
    if enable_mlflow:
        mlflow.log_param("checkpoint_prefix", checkpoint_prefix)
    best_env_reward = float("-inf")
    latest_sac_validation = None

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
        "policy_activation": str(policy_cfg.get("activation", "relu")),
        "normalize_observations": observation_norm_kwargs["normalize_observations"],
        "observation_norm_epsilon": observation_norm_kwargs["observation_norm_epsilon"],
        "observation_norm_clip": observation_norm_kwargs["observation_norm_clip"],
        "init_policy_from_expert": init_from_expert,
        "action_low": env.action_low.tolist(),
        "action_high": env.action_high.tolist(),
        "method": gradient_method,
        "estimator_weighting": estimator_weighting,
        "agent": "sac",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    def log_reward_stats(outer_step: int, agent_trajs, rank_corr_val: float) -> None:
        if estimator_weighting == "uniform_observations":
            reward_stats = {
                "expert": learned_reward_observation_stats(reward, expert_valid_trajs),
                "random": learned_reward_observation_stats(reward, random_valid_trajs),
                "agent": learned_reward_observation_stats(reward, agent_trajs),
            }

            metrics = {"reward/rank_corr": rank_corr_val}
            for split_name, stats in reward_stats.items():
                metrics.update(
                    {
                        f"reward/{split_name}/{key}": value
                        for key, value in stats.items()
                    }
                )
            metrics["reward/expert_agent_observation_gap"] = (
                reward_stats["expert"]["observation_mean"]
                - reward_stats["agent"]["observation_mean"]
            )

            logger.info(
                "   [reward] "
                f"rank_corr={rank_corr_val:.3f} "
                f"expert_observation={reward_stats['expert']['observation_mean']:.3f}"
                f"+/-{reward_stats['expert']['observation_std']:.3f} "
                f"random_observation={reward_stats['random']['observation_mean']:.3f}"
                f"+/-{reward_stats['random']['observation_std']:.3f} "
                f"agent_observation={reward_stats['agent']['observation_mean']:.3f}"
                f"+/-{reward_stats['agent']['observation_std']:.3f} "
                f"counts=({int(reward_stats['expert']['observation_count'])},"
                f"{int(reward_stats['random']['observation_count'])},"
                f"{int(reward_stats['agent']['observation_count'])})"
            )

            if enable_mlflow:
                mlflow.log_metrics(metrics, step=outer_step)
            return

        reward_stats = {
            "expert": learned_reward_stats(reward, expert_valid_trajs, outer_optimizer.discount),
            "random": learned_reward_stats(reward, random_valid_trajs, outer_optimizer.discount),
            "agent": learned_reward_stats(reward, agent_trajs, outer_optimizer.discount),
        }

        metrics = {"reward/rank_corr": rank_corr_val}
        for split_name, stats in reward_stats.items():
            metrics.update(
                {
                    f"reward/{split_name}/{key}": value
                    for key, value in stats.items()
                    if key.startswith("return_") or key.startswith("step_")
                }
            )

        logger.info(
            "   [reward] "
            f"rank_corr={rank_corr_val:.3f} "
            f"expert_return={reward_stats['expert']['return_mean']:.3f}"
            f"+/-{reward_stats['expert']['return_std']:.3f} "
            f"random_return={reward_stats['random']['return_mean']:.3f}"
            f"+/-{reward_stats['random']['return_std']:.3f} "
            f"agent_return={reward_stats['agent']['return_mean']:.3f}"
            f"+/-{reward_stats['agent']['return_std']:.3f} "
            f"expert_q50={reward_stats['expert']['return_q50']:.3f} "
            f"random_q50={reward_stats['random']['return_q50']:.3f} "
            f"agent_q50={reward_stats['agent']['return_q50']:.3f}"
        )

        if enable_mlflow:
            mlflow.log_metrics(metrics, step=outer_step)

    def log_and_checkpoint(outer_step: int, agent_trajs):
        nonlocal best_env_reward

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm
        parameter_update_norm = outer_optimizer.parameter_update_norm
        fisher_parameter_update_norm = (
            parameter_update_norm if gradient_method == "fisher" else float("nan")
        )
        ml_counterfactual_update_norm = outer_optimizer.ml_counterfactual_update_norm
        ml_irl_update_direction_norm = outer_optimizer.ml_irl_update_direction_norm
        fisher_ml_cosine = outer_optimizer.fisher_ml_cosine

        if estimator_weighting == "uniform_observations":
            l_outer = observation_outer_loss(policy, expert_train_trajs)
        else:
            l_outer = outer_loss(policy, expert_train_trajs, outer_optimizer.discount)

        agent_len = mean_trajectory_length(agent_trajs)
        expert_len = mean_trajectory_length(expert_train_trajs)

        agent_ret = mean_trajectory_return(agent_trajs)
        expert_ret = mean_trajectory_return(expert_train_trajs)
        validation_ret = agent_ret
        if (
            save_best_checkpoint
            and latest_sac_validation is not None
            and latest_sac_validation["outer_step"] == outer_step
        ):
            validation_ret = latest_sac_validation["agent_return"]

        rank_corr_val = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
        policy_nll_val = policy_nll(policy, expert_valid_trajs)
        policy_entropy_val = policy_entropy(policy, agent_trajs)
        log_reward_stats(outer_step, agent_trajs, rank_corr_val)

        history["l_outer"].append(l_outer)
        history["agent_len"].append(agent_len)
        history["expert_len"].append(expert_len)
        history["agent_return"].append(agent_ret)
        history["expert_return"].append(expert_ret)
        history["rank_corr"].append(rank_corr_val)
        history["policy_nll"].append(policy_nll_val)
        history["raw_hypgrad_norm"].append(raw_hypgrad_norm)
        history["clipped_hypgrad_norm"].append(clipped_hypgrad_norm)
        history["reward/parameter_update_norm"].append(parameter_update_norm)
        history["reward/fisher_parameter_update_norm"].append(fisher_parameter_update_norm)
        history["reward/ml_counterfactual_update_norm"].append(ml_counterfactual_update_norm)
        history["policy_entropy"].append(policy_entropy_val)
        history["ml_irl_update_direction_norm"].append(ml_irl_update_direction_norm)
        history["fisher_ml_cosine"].append(fisher_ml_cosine)
        history["lr_outer"].append(lr_outer_current)

        is_best_checkpoint = validation_ret > best_env_reward
        if is_best_checkpoint:
            best_env_reward = validation_ret

        checkpoint_path = ckpt_dir / f"{checkpoint_prefix}_outer_{outer_step:04d}.pt"
        save_checkpoint(
            path=str(checkpoint_path),
            policy=policy,
            reward=reward,
            arch=arch,
            outer_step=outer_step,
            best_env_reward=best_env_reward,
            validation_return=validation_ret,
            selection_metric="validation/agent_return",
            run_identifier=checkpoint_run_id,
        )
        if enable_mlflow:
            log_artifact_if_exists(checkpoint_path, artifact_path="checkpoints")

        best_checkpoint_path = None
        if is_best_checkpoint and save_best_checkpoint:
            best_checkpoint_path = ckpt_dir / f"{checkpoint_prefix}_best.pt"
            save_checkpoint(
                path=str(best_checkpoint_path),
                policy=policy,
                reward=reward,
                arch=arch,
                outer_step=outer_step,
                best_env_reward=best_env_reward,
                validation_return=validation_ret,
                selection_metric="validation/agent_return",
                run_identifier=checkpoint_run_id,
            )
            logger.info(
                "Best validation checkpoint updated | "
                f"outer_step={outer_step} | return={validation_ret:.3f} | "
                f"path={best_checkpoint_path}"
            )

        if is_best_checkpoint and enable_mlflow:
            mlflow.set_tag("best_checkpoint", checkpoint_path.name)

        if metrics_callback is not None:
            metrics_callback(
                outer_step,
                {
                    "validation/agent_return": validation_ret,
                    "validation/best_agent_return": best_env_reward,
                    "validation/expert_return": expert_ret,
                    "outer/loss": l_outer,
                    "reward/rank_corr": rank_corr_val,
                    "gradient/fisher_ml_cosine": fisher_ml_cosine,
                    "best_checkpoint_path": (
                        str(best_checkpoint_path)
                        if best_checkpoint_path is not None
                        else None
                    ),
                },
            )

        row = (
            f"{outer_step:>5} | {l_outer:>10.3f} | {agent_len:>10.1f} | "
            f"{expert_len:>10.1f} | {agent_ret:>10.1f} | {expert_ret:>10.1f} | "
            f"{rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{parameter_update_norm:>11.3e} | {ml_counterfactual_update_norm:>11.3e} | "
            f"{policy_entropy_val:>10.3f} | "
            f"{ml_irl_update_direction_norm:>10.3f} | {fisher_ml_cosine:>9.4f} | "
            f"{lr_outer_current:>12.2e}"
        )

        logger.info(row)

        if enable_mlflow:
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
                    "policy_entropy": policy_entropy_val,
                    "lr_reward": lr_outer_current,
                },
                step=outer_step,
            )

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'RewardStep':>11} | {'MLStep':>11} | {'PolicyEnt':>10} | "
        f"{'MLNorm':>10} | {'GradCos':>9} | "
        f"{'lr_outer':>12}"
    )

    logger.info(header)

    # eval_reward_stats()

    policy_viz_disabled = False

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

    sac_train_env = Environment(id=config["env"]["id"], seed=int(inner_cfg["eval_env_seed"]))
    sac_eval_env = Environment(
        id=config["env"]["id"],
        seed=int(inner_cfg["train_env_seed"]),
        custom_reward_fn=None,
    )

    def inner_optimize(outer_step: int) -> float:
        nonlocal latest_sac_validation
        logger.info(
            f"SAC inner phase | outer_step={outer_step} | steps={n_inner_steps} | "
            f"replay_size={len(sac.replay_buffer)} | total_env_steps={sac.total_env_steps} | "
            f"total_gradient_updates={sac.total_gradient_updates}"
        )

        @torch.no_grad()
        def validate(ts: int):
            nonlocal latest_sac_validation
            sac.policy.eval()

            agent_valid_trajs = collect_trajectories(
                env=sac_eval_env,
                policy=sac.policy,
                n=int(sac_cfg.get("n_eval_traj", 10)),
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

            if estimator_weighting == "uniform_observations":
                l_outer = observation_outer_loss(sac.policy, expert_valid_trajs)
            else:
                l_outer = outer_loss(
                    policy=sac.policy,
                    expert_trajs=expert_valid_trajs,
                    discount=sac.gamma,
                    normalize_discounted_sums=outer_optimizer.normalize_discounted_sums,
                )
            validation_agent_return = mean_trajectory_return(agent_valid_trajs)
            validation_agent_length = mean_trajectory_length(agent_valid_trajs)
            latest_sac_validation = {
                "outer_step": outer_step,
                "agent_return": validation_agent_return,
                "agent_length": validation_agent_length,
            }

            if enable_mlflow:
                mlflow_step = sac.total_env_steps
                mlflow.log_metrics(
                    {
                        "sac/l_inner": float(l_inner),
                        "sac/l_outer": float(l_outer),
                        "sac/agent_return": validation_agent_return,
                        "sac/agent_len": validation_agent_length,
                        "sac/policy_entropy": policy_entropy(sac.policy, agent_valid_trajs),
                        "sac/outer_step": float(outer_step),
                        "sac/env_step": float(sac.total_env_steps),
                        "sac/phase_env_step": float(ts),
                        "sac/replay_size": float(len(sac.replay_buffer)),
                        "sac/gradient_updates": float(sac.total_gradient_updates),
                    },
                    step=mlflow_step,
                )

            sac.policy.train()

        inner_started = profiling_time()
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
            reward_fn=reward.rewards,
            validate_fn=validate,
            validate_every=int(sac_cfg["validate_every"]),
        )
        inner_loop_seconds = profiling_time() - inner_started
        logger.info(
            f"SAC inner phase complete | outer_step={outer_step} | "
            f"elapsed={inner_loop_seconds:.3f}s"
        )
        return inner_loop_seconds

    initial_outer_started = profiling_time()
    inner_loop_seconds = inner_optimize(outer_step=0)

    def collect_outer_trajectories():
        n_outer_traj = (
            max(n_agent_traj, n_fisher_traj)
            if gradient_method == "fisher"
            else n_agent_traj
        )
        outer_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_outer_traj,
            max_steps=int(config["env"]["max_steps"]),
            desc=(
                "agent/Fisher outer trajs"
                if gradient_method == "fisher"
                else "ML-IRL agent outer trajs"
            ),
        )
        fisher_trajs = outer_trajs[:n_fisher_traj] if gradient_method == "fisher" else []
        return outer_trajs[:n_agent_traj], fisher_trajs

    collection_started = profiling_time()
    agent_trajs, fisher_trajs = collect_outer_trajectories()
    trajectory_collection_seconds = profiling_time() - collection_started

    evaluation_checkpoint_started = profiling_time()
    log_and_checkpoint(outer_step=0, agent_trajs=agent_trajs)
    evaluation_checkpoint_seconds = profiling_time() - evaluation_checkpoint_started
    initial_outer_step_seconds = profiling_time() - initial_outer_started
    record_timing(
        outer_step=0,
        inner_loop_seconds=inner_loop_seconds,
        reward_update_seconds=0.0,
        trajectory_collection_seconds=trajectory_collection_seconds,
        visualization_seconds=0.0,
        evaluation_checkpoint_seconds=evaluation_checkpoint_seconds,
        outer_step_seconds=initial_outer_step_seconds,
    )

    for outer_step in range(1, n_outer_steps + 1):
        outer_step_started = profiling_time()
        visualization_seconds = 0.0
        should_visualize = (
            (outer_step - 1) % visualization_traj_frequency == 0
        )
        if should_visualize and not policy_viz_disabled:
            visualization_started = profiling_time()
            policy_viz_disabled = not log_policy_trajectory_gifs(
                env_id=config["env"]["id"],
                policy=policy,
                outer_step=outer_step,
                max_steps=int(config["env"]["max_steps"]),
                output_root=Path(log_cfg["report_dir"]),
                cfg=fisher_cfg.get("visualization", {}),
                seed_base=int(fisher_cfg["env_seed"]) + 10_000,
                logger=logger,
            )
            visualization_seconds = profiling_time() - visualization_started

        reward_update_started = profiling_time()
        outer_optimizer.step(expert_train_trajs, agent_trajs, fisher_trajs)
        reward_update_seconds = profiling_time() - reward_update_started
        # eval_reward_stats()

        inner_loop_seconds = inner_optimize(outer_step=outer_step)

        collection_started = profiling_time()
        agent_trajs, fisher_trajs = collect_outer_trajectories()
        trajectory_collection_seconds = profiling_time() - collection_started

        evaluation_checkpoint_started = profiling_time()
        log_and_checkpoint(outer_step=outer_step, agent_trajs=agent_trajs)
        evaluation_checkpoint_seconds = profiling_time() - evaluation_checkpoint_started
        outer_step_seconds = profiling_time() - outer_step_started
        record_timing(
            outer_step=outer_step,
            inner_loop_seconds=inner_loop_seconds,
            reward_update_seconds=reward_update_seconds,
            trajectory_collection_seconds=trajectory_collection_seconds,
            visualization_seconds=visualization_seconds,
            evaluation_checkpoint_seconds=evaluation_checkpoint_seconds,
            outer_step_seconds=outer_step_seconds,
        )

    sac_train_env.close()
    sac_eval_env.close()
    env.close()
    
    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD or ML-IRL SAC — Hopper")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--estimator-weighting",
        choices=("discounted_trajectories", "uniform_observations"),
        default=None,
        help="Override how outer-estimator timesteps are weighted and averaged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("hopper", args.config)
    config = load_config(config_path)

    if args.estimator_weighting is not None:
        config["fisher"]["estimator_weighting"] = args.estimator_weighting

    reward_max_grad_norm_override = os.environ.get("REWARD_MAX_GRAD_NORM")
    if reward_max_grad_norm_override is not None:
        reward_max_grad_norm = float(reward_max_grad_norm_override)
        if reward_max_grad_norm <= 0.0:
            raise ValueError("REWARD_MAX_GRAD_NORM must be greater than zero.")
        config["fisher"]["max_grad_norm"] = reward_max_grad_norm

    run_output_suffix = os.environ.get("RUN_OUTPUT_SUFFIX")
    if run_output_suffix:
        if Path(run_output_suffix).name != run_output_suffix or run_output_suffix in {".", ".."}:
            raise ValueError("RUN_OUTPUT_SUFFIX must be a single safe path component.")
        config["logging"]["report_dir"] = str(
            Path(config["logging"]["report_dir"]) / run_output_suffix
        )
        config["checkpoint"]["dir"] = str(
            Path(config["checkpoint"]["dir"]) / run_output_suffix
        )

    log_cfg = config["logging"]
    gradient_method = str(config["fisher"].get("gradient_method", "fisher")).lower().replace("-", "_")
    estimator_weighting = str(
        config["fisher"].get("estimator_weighting", "discounted_trajectories")
    ).lower().replace("-", "_")
    run_label = (
        gradient_method
        if estimator_weighting == "discounted_trajectories"
        else f"{gradient_method}_{estimator_weighting}"
    )

    log_dir = log_cfg["log_dir"]
    logger_name = f"{run_label}_hopper"
    if run_output_suffix:
        logger_name = f"{logger_name}_{run_output_suffix}"
    logger = get_logger(logger_name, log_dir=log_dir)

    logger.info(
        f"=== {gradient_method} Hopper SAC | estimator_weighting={estimator_weighting} ==="
    )
    if reward_max_grad_norm_override is not None:
        logger.info(f"Reward gradient clip override: {config['fisher']['max_grad_norm']}")
    if run_output_suffix:
        logger.info(f"Run output suffix: {run_output_suffix}")

    with begin_mlflow_run(
        config,
        config_path,
        method=run_label,
        env_name="hopper",
        agent="sac",
    ):
        history = train_bilevel(config, logger)

        report_path = Path(log_cfg["report_dir"]) / f"{run_label}_sac_hopper_history.json"
        save_history(history, str(report_path))
        log_artifact_if_exists(report_path, artifact_path="reports")
        logger.info(f"History saved to {report_path}")


if __name__ == "__main__":
    main()
