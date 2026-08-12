import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class LQR:
    def __init__(
        self,
        *,
        A,
        B,
        Q,
        R,
        process_cov,
        initial_cov,
        seed: int,
        max_episode_steps: int = 100,
        action_limit: float = 5.0,
        observation_limit: float = 100.0,
        termination_state_norm: float = 50.0,
        reward_scale: float = 1.0,
        custom_reward_fn=None,
    ):
        self.seed = seed

        self.A = torch.as_tensor(A, dtype=torch.float32)
        self.B = torch.as_tensor(B, dtype=torch.float32)
        self.Q = torch.as_tensor(Q, dtype=torch.float32)
        self.R = torch.as_tensor(R, dtype=torch.float32)
        self.process_cov = torch.as_tensor(process_cov, dtype=torch.float32)
        self.initial_cov = torch.as_tensor(initial_cov, dtype=torch.float32)

        self.max_episode_steps = int(max_episode_steps)
        self.action_limit = float(action_limit)
        self.observation_limit = float(observation_limit)
        self.termination_state_norm = float(termination_state_norm)
        self.reward_scale = float(reward_scale)
        self.custom_reward_fn = custom_reward_fn

        self._validate_parameters()

        self.state_dim = self.A.shape[0]
        self.action_dim = self.B.shape[1]

        self.is_discrete = False
        self.is_continuous = True

        self.action_low = torch.full(size=(self.action_dim,), fill_value=-self.action_limit, dtype=torch.float32)
        self.action_high = torch.full(size=(self.action_dim,), fill_value=self.action_limit, dtype=torch.float32)

        self.observation_low = torch.full(size=(self.state_dim,), fill_value=-self.observation_limit, dtype=torch.float32)
        self.observation_high = torch.full(size=(self.state_dim,), fill_value=self.observation_limit, dtype=torch.float32)

        self.action_space = spaces.Box(
            low=self.action_low.detach().cpu().numpy(),
            high=self.action_high.detach().cpu().numpy(),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=self.observation_low.detach().cpu().numpy(),
            high=self.observation_high.detach().cpu().numpy(),
            dtype=np.float32,
        )

        self._process_cov_sqrt = self._covariance_sqrt(self.process_cov)
        self._initial_cov_sqrt = self._covariance_sqrt(self.initial_cov)

        self.generator = self._make_generator(self.seed)
        self.state = torch.zeros(self.state_dim, dtype=torch.float32)
        self.elapsed_steps = 0

        self.reset()

    def _make_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _is_symmetric(matrix: torch.Tensor, tolerance: float = 1e-6) -> bool:
        return torch.allclose(matrix, matrix.mT, atol=tolerance, rtol=tolerance)

    @staticmethod
    def _minimum_eigenvalue(matrix: torch.Tensor) -> float:
        symmetric_matrix = 0.5 * (matrix + matrix.mT)
        return torch.linalg.eigvalsh(symmetric_matrix).min().item()

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

        for name, expected_shape in expected_shapes.items():
            value = getattr(self, name)

            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, " f"got {tuple(value.shape)}.")

            if not self._is_symmetric(value):
                raise ValueError(f"{name} must be symmetric.")

        eigenvalue_tolerance = 1e-6

        if self._minimum_eigenvalue(self.Q) < -eigenvalue_tolerance:
            raise ValueError("Q must be positive semidefinite.")

        if self._minimum_eigenvalue(self.R) <= 0.0:
            raise ValueError("R must be positive definite.")

        if self._minimum_eigenvalue(self.process_cov) < -eigenvalue_tolerance:
            raise ValueError("process_cov must be positive semidefinite.")

        if self._minimum_eigenvalue(self.initial_cov) < -eigenvalue_tolerance:
            raise ValueError("initial_cov must be positive semidefinite.")

        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")

        if self.action_limit <= 0.0:
            raise ValueError("action_limit must be positive.")

        if self.observation_limit <= 0.0:
            raise ValueError("observation_limit must be positive.")

        if not (0.0 < self.termination_state_norm <= self.observation_limit):
            raise ValueError("termination_state_norm must lie in " "(0, observation_limit].")

        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")

    def _covariance_sqrt(self, covariance: torch.Tensor) -> torch.Tensor:
        covariance = 0.5 * (covariance + covariance.mT)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp_min(0.0)

        return eigenvectors @ torch.diag(eigenvalues.sqrt())

    def _sample_gaussian(self, covariance_sqrt: torch.Tensor) -> torch.Tensor:
        standard_normal = torch.randn(self.state_dim, dtype=torch.float32, generator=self.generator)
        return covariance_sqrt @ standard_normal

    def clone(self):
        return LQR(
            A=self.A.clone(),
            B=self.B.clone(),
            Q=self.Q.clone(),
            R=self.R.clone(),
            process_cov=self.process_cov.clone(),
            initial_cov=self.initial_cov.clone(),
            seed=self.seed,
            max_episode_steps=self.max_episode_steps,
            action_limit=self.action_limit,
            observation_limit=self.observation_limit,
            termination_state_norm=self.termination_state_norm,
            reward_scale=self.reward_scale,
            custom_reward_fn=self.custom_reward_fn,
        )

    def reset(self, *, seed: int = None, options=None) -> torch.Tensor:
        if seed is not None:
            self.seed = int(seed)
            self.generator = self._make_generator(self.seed)

        options = options or {}

        if "state" in options:
            state = torch.as_tensor(options["state"], dtype=torch.float32).flatten()

            if state.shape != (self.state_dim,):
                raise ValueError("reset state must have shape " f"({self.state_dim},), got {tuple(state.shape)}.")

            self.state = state.clone()
        else:
            self.state = self._sample_gaussian(self._initial_cov_sqrt)

        self.state = self.state.clamp(min=-self.observation_limit, max=self.observation_limit)
        self.elapsed_steps = 0
        return self.state.clone()

    def step(self, action: torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32).flatten()

        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), " f"got {tuple(action.shape)}.")

        bounded_action = action.clamp(min=-self.action_limit, max=self.action_limit)

        current_state = self.state
        state_cost = current_state @ self.Q @ current_state
        action_cost = bounded_action @ self.R @ bounded_action
        environment_reward = -0.5 * self.reward_scale * (state_cost + action_cost)

        if self.custom_reward_fn is None:
            reward = environment_reward
        else:
            reward = self.custom_reward_fn(current_state, bounded_action)
            reward = torch.as_tensor(reward, dtype=torch.float32)

        process_noise = self._sample_gaussian(self._process_cov_sqrt)
        next_state = self.A @ current_state + self.B @ bounded_action + process_noise
        self.elapsed_steps += 1
        state_norm = torch.linalg.vector_norm(next_state)

        terminated = ~torch.isfinite(state_norm) | (state_norm >= self.termination_state_norm)
        truncated = torch.tensor(self.elapsed_steps >= self.max_episode_steps, dtype=torch.bool)

        next_state = torch.nan_to_num(
            next_state,
            nan=0.0,
            posinf=self.observation_limit,
            neginf=-self.observation_limit,
        )
        next_state = next_state.clamp(min=-self.observation_limit, max=self.observation_limit)

        self.state = next_state

        return (
            next_state.clone(),
            reward.reshape(1),
            terminated.reshape(1),
            truncated.reshape(1),
        )

    def get_random_action(self) -> torch.Tensor:
        random_values = torch.rand(self.action_dim, dtype=torch.float32, generator=self.generator)
        return self.action_low + random_values * (self.action_high - self.action_low)

    def environment_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = torch.as_tensor(state, dtype=torch.float32).flatten()
        action = torch.as_tensor(action, dtype=torch.float32).flatten()

        if state.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},), " f"got {tuple(state.shape)}.")

        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), " f"got {tuple(action.shape)}.")

        action = action.clamp(min=-self.action_limit, max=self.action_limit)
        state_cost = state @ self.Q @ state
        action_cost = action @ self.R @ action
        return (-0.5 * self.reward_scale * (state_cost + action_cost)).reshape(1)

    def close(self) -> None:
        pass


class GymLQR(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, torch_env: LQR, render_mode: str | None = None):
        super().__init__()

        if render_mode is not None:
            raise ValueError("LQR environment does not support rendering.")

        self.torch_env = torch_env
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-torch_env.observation_limit,
            high=torch_env.observation_limit,
            shape=(torch_env.state_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-torch_env.action_limit,
            high=torch_env.action_limit,
            shape=(torch_env.action_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        state = self.torch_env.reset(seed=seed, options=options)
        observation = state.detach().cpu().numpy().astype(np.float32)
        return observation, {}

    def step(self, action):
        action_tensor = torch.as_tensor(action, dtype=torch.float32)

        with torch.no_grad():
            next_state, reward, terminated, truncated = self.torch_env.step(action_tensor)

        observation = next_state.detach().cpu().numpy().astype(np.float32)
        info = {}

        return (
            observation,
            float(reward.item()),
            bool(terminated.item()),
            bool(truncated.item()),
            info,
        )

    def render(self):
        return None

    def close(self):
        self.torch_env.close()
