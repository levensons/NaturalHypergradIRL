"""Gymnasium environment for the neural-SAC LQR benchmark."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

DEFAULT_ENV_ID = "NeuralLQR-v0"


class LinearQuadraticEnv(gym.Env):
    """Finite episodes from stationary linear dynamics and quadratic cost.

    The environment reward is the negative ground-truth cost

        -reward_scale / 2 * (x.T Q x + u.T R u).

    Actions are bounded because the benchmark deliberately uses the same
    tanh-squashed policy as Hopper. This makes it a bounded LQR task rather
    than the exact unconstrained problem solved by Riccati recursion.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        A,
        B,
        Q,
        R,
        process_cov,
        initial_cov,
        horizon: int = 100,
        action_limit: float = 5.0,
        observation_limit: float = 100.0,
        termination_state_norm: float = 50.0,
        reward_scale: float = 1.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        if render_mode is not None:
            raise ValueError("LinearQuadraticEnv does not support rendering.")

        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.process_cov = np.asarray(process_cov, dtype=np.float64)
        self.initial_cov = np.asarray(initial_cov, dtype=np.float64)
        self.horizon = int(horizon)
        self.action_limit = float(action_limit)
        self.observation_limit = float(observation_limit)
        self.termination_state_norm = float(termination_state_norm)
        self.reward_scale = float(reward_scale)

        self._validate_parameters()
        self.state_dim = self.A.shape[0]
        self.action_dim = self.B.shape[1]
        self.action_space = spaces.Box(
            low=-self.action_limit,
            high=self.action_limit,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-self.observation_limit,
            high=self.observation_limit,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.state = np.zeros(self.state_dim, dtype=np.float64)
        self.elapsed_steps = 0

    def _validate_parameters(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square.")
        state_dim = self.A.shape[0]
        if self.B.ndim != 2 or self.B.shape[0] != state_dim:
            raise ValueError("B must have shape (state_dim, action_dim).")
        action_dim = self.B.shape[1]
        expected_shapes = {
            "Q": (state_dim, state_dim),
            "R": (action_dim, action_dim),
            "process_cov": (state_dim, state_dim),
            "initial_cov": (state_dim, state_dim),
        }
        for name, shape in expected_shapes.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}.")
            if not np.allclose(value, value.T):
                raise ValueError(f"{name} must be symmetric.")

        if np.linalg.eigvalsh(self.Q).min() < 0.0:
            raise ValueError("Q must be positive semidefinite.")
        if np.linalg.eigvalsh(self.R).min() <= 0.0:
            raise ValueError("R must be positive definite.")
        if np.linalg.eigvalsh(self.process_cov).min() < -1e-12:
            raise ValueError("process_cov must be positive semidefinite.")
        if np.linalg.eigvalsh(self.initial_cov).min() < -1e-12:
            raise ValueError("initial_cov must be positive semidefinite.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.action_limit <= 0.0 or self.observation_limit <= 0.0:
            raise ValueError("action and observation limits must be positive.")
        if not 0.0 < self.termination_state_norm <= self.observation_limit:
            raise ValueError(
                "termination_state_norm must lie in (0, observation_limit]."
            )
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        options = options or {}
        if "state" in options:
            state = np.asarray(options["state"], dtype=np.float64)
            if state.shape != (self.state_dim,):
                raise ValueError(
                    f"reset state must have shape ({self.state_dim},), got {state.shape}."
                )
            self.state = state.copy()
        else:
            self.state = self.np_random.multivariate_normal(
                np.zeros(self.state_dim, dtype=np.float64), self.initial_cov
            )
        self.state = np.clip(
            self.state, -self.observation_limit, self.observation_limit
        )
        self.elapsed_steps = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"action must have shape ({self.action_dim},), got {action.shape}."
            )
        action = np.clip(action, -self.action_limit, self.action_limit)

        state_cost = float(self.state @ self.Q @ self.state)
        action_cost = float(action @ self.R @ action)
        reward = -0.5 * self.reward_scale * (state_cost + action_cost)
        process_noise = self.np_random.multivariate_normal(
            np.zeros(self.state_dim, dtype=np.float64), self.process_cov
        )
        next_state = self.A @ self.state + self.B @ action + process_noise
        self.elapsed_steps += 1

        state_norm = float(np.linalg.norm(next_state))
        terminated = (
            not np.isfinite(state_norm) or state_norm >= self.termination_state_norm
        )
        truncated = self.elapsed_steps >= self.horizon
        self.state = np.nan_to_num(
            next_state,
            nan=0.0,
            posinf=self.observation_limit,
            neginf=-self.observation_limit,
        )
        self.state = np.clip(
            self.state, -self.observation_limit, self.observation_limit
        )
        observation = self.state.astype(np.float32)
        info = {
            "state_cost": state_cost,
            "action_cost": action_cost,
            "state_norm": state_norm,
        }
        return observation, float(reward), terminated, truncated, info


def register_lqr_env(config: dict, env_id: str | None = None) -> str:
    """Register (or refresh) the configured LQR environment in Gymnasium."""
    env_id = env_id or config.get("env", {}).get("id", DEFAULT_ENV_ID)
    lqr_cfg = config["lqr"]
    kwargs = {
        key: lqr_cfg[key]
        for key in (
            "A",
            "B",
            "Q",
            "R",
            "process_cov",
            "initial_cov",
            "horizon",
            "action_limit",
            "observation_limit",
            "termination_state_norm",
            "reward_scale",
        )
    }
    if env_id in gym.registry:
        del gym.registry[env_id]
    gym.register(id=env_id, entry_point=LinearQuadraticEnv, kwargs=kwargs)
    return env_id
