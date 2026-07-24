from collections.abc import Callable

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.env import Environment
from src.utils.policies import Policy


def observation_normalization_kwargs(sac_config: dict) -> dict:
    """Translate the optional SAC observation-normalization config for Policy."""
    config = sac_config.get("observation_normalization", {})
    if isinstance(config, bool):
        config = {"enabled": config}
    if not isinstance(config, dict):
        raise TypeError("observation_normalization must be a mapping or boolean.")

    epsilon = float(config.get("epsilon", 1e-8))
    clip = config.get("clip", 10.0)
    clip = None if clip is None else float(clip)
    if epsilon <= 0.0:
        raise ValueError("observation_normalization.epsilon must be positive.")
    if clip is not None and clip <= 0.0:
        raise ValueError("observation_normalization.clip must be positive or null.")

    return {
        "normalize_observations": bool(config.get("enabled", False)),
        "observation_norm_epsilon": epsilon,
        "observation_norm_clip": clip,
    }


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_buf = torch.empty(capacity, state_dim, dtype=torch.float32)
        self.action_buf = torch.empty(capacity, action_dim, dtype=torch.float32)
        self.reward_buf = torch.empty(capacity, 1, dtype=torch.float32)
        self.next_state_buf = torch.empty(capacity, state_dim, dtype=torch.float32)
        self.done_buf = torch.empty(capacity, 1, dtype=torch.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, done):
        self.state_buf[self.ptr] = torch.as_tensor(state, dtype=torch.float32).detach().cpu()
        self.action_buf[self.ptr] = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
        self.reward_buf[self.ptr] = torch.as_tensor(reward, dtype=torch.float32).detach().cpu().reshape(1)
        self.next_state_buf[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32).detach().cpu()
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32).detach().cpu().reshape(1)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str | None = None):
        idxs = torch.randint(0, self.size, size=(batch_size,))

        states = self.state_buf[idxs]
        actions = self.action_buf[idxs]
        rewards = self.reward_buf[idxs].reshape(-1)
        next_states = self.next_state_buf[idxs]
        dones = self.done_buf[idxs].reshape(-1)

        if device is not None:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

        return states, actions, rewards, next_states, dones
    
    def recalc_rewards(self, reward_fn, batch_size: int = 4096):
        if self.size == 0:
            return
        
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)

            states = self.state_buf[start:end]
            actions = self.action_buf[start:end]

            rewards = reward_fn(states, actions)
            self.reward_buf[start:end, :] = torch.as_tensor(rewards, dtype=torch.float32).detach().cpu().reshape(-1, 1)


class QFunction(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 1,
    ):
        super().__init__()

        in_dim = state_dim + action_dim
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]

        self.backbone = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (B, state_dim)
        # actions: (B, action_dim)
        x = torch.cat([states, actions], dim=-1)
        out = self.backbone(x).squeeze(-1)
        return out  # (B,)


class SAC:
    def __init__(
        self,
        policy: Policy,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 1,
        gamma: float = 0.99,
        alpha: float = 1.0,
        tau: float = 0.005,
        replay_buffer_capacity: int = 1_000_000,
    ):
        self.policy = policy
        self.device = next(policy.parameters()).device

        self.q1 = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers).to(self.device)
        self.q2 = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers).to(self.device)

        self.q1_target = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())

        self.q2_target = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers).to(self.device)
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, capacity=replay_buffer_capacity)

        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

        self.policy_params = list(self.policy.parameters())
        self.critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.policy_optimizer = None
        self.critic_optimizer = None
        self.total_env_steps = 0
        self.total_gradient_updates = 0

    def _update_observation_stats(self, observations: torch.Tensor) -> None:
        update = getattr(self.policy, "update_observation_stats", None)
        if update is not None:
            update(observations)

    def _normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        normalize = getattr(self.policy, "normalize_observations", None)
        if normalize is None:
            return observations
        return normalize(observations)

    def _ensure_optimizers(self, actor_lr: float, critic_lr: float) -> None:
        if self.policy_optimizer is None:
            self.policy_optimizer = torch.optim.Adam(self.policy_params, lr=actor_lr)
        else:
            for param_group in self.policy_optimizer.param_groups:
                param_group["lr"] = actor_lr

        if self.critic_optimizer is None:
            self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=critic_lr)
        else:
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = critic_lr

    @torch.no_grad()
    def soft_update(self):
        for tp, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.mul_(1.0 - self.tau)
            tp.add_(self.tau * p)

        for tp, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.mul_(1.0 - self.tau)
            tp.add_(self.tau * p)

    def optimize(
        self,
        train_env: Environment,
        total_steps: int = 1_000_000,
        learning_starts: int = 10_000,
        batch_size: int = 256,
        max_grad_norm: float | None = 1.0,
        gradient_update_steps: int = 2,
        target_update_interval: int = 1,
        critic_lr: float = 1e-3,
        actor_lr: float = 3e-4,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        validate_fn = None,
        validate_every: int = 1000,
    ):
        state = train_env.reset()
        self.device = next(self.policy.parameters()).device
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.q1_target.to(self.device)
        self.q2_target.to(self.device)

        self._ensure_optimizers(actor_lr=actor_lr, critic_lr=critic_lr)

        tqdm.write(
            "SAC budget | "
            "envs=1 | "
            f"steps={total_steps} | "
            f"updates_per_step={gradient_update_steps}"
        )

        for step in tqdm(range(total_steps), desc="SAC inner optimization", leave=False):
            # COLLECTING
            self._update_observation_stats(state)
            if len(self.replay_buffer) < learning_starts:
                action = train_env.get_random_action()
            else:
                with torch.no_grad():
                    action = self.policy.sample(state.to(self.device))

            next_state, reward, done = train_env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            self.total_env_steps += 1

            if done:
                state = train_env.reset()
            else:
                state = next_state

            if len(self.replay_buffer) < learning_starts:
                continue

            # UPDATE
            for _ in range(gradient_update_steps):
                # CRITIC
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, device=self.device)
                normalized_states = self._normalize_observations(states)
                normalized_next_states = self._normalize_observations(next_states)

                with torch.no_grad():
                    if reward_fn is not None:
                        rewards = reward_fn(states, actions).to(self.device).reshape(-1)

                    next_actions, next_log_probs = self.policy.sample(next_states, return_log_probs=True)

                    next_q1 = self.q1_target(normalized_next_states, next_actions)
                    next_q2 = self.q2_target(normalized_next_states, next_actions)
                    next_q = torch.min(next_q1, next_q2)
                    target_q = rewards + self.gamma * (1.0 - dones) * (next_q - self.alpha * next_log_probs)

                current_q1 = self.q1(normalized_states, actions)
                current_q2 = self.q2(normalized_states, actions)

                q1_loss = F.mse_loss(current_q1, target_q)
                q2_loss = F.mse_loss(current_q2, target_q)

                critic_loss = 0.5 * (q1_loss + q2_loss)

                self.critic_optimizer.zero_grad()
                torch.autograd.backward(critic_loss, inputs=self.critic_params)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.critic_params, max_grad_norm)
                self.critic_optimizer.step()

                # ACTOR
                new_actions, log_probs = self.policy.sample(states, return_log_probs=True)

                q1 = self.q1(normalized_states, new_actions)
                q2 = self.q2(normalized_states, new_actions)
                q = torch.min(q1, q2)

                policy_loss = torch.mean(self.alpha * log_probs - q)

                self.policy_optimizer.zero_grad()
                torch.autograd.backward(policy_loss, inputs=self.policy_params)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy_params, max_grad_norm)
                self.policy_optimizer.step()

                # CRITIC-TARGET SOFT-UPDATE
                self.total_gradient_updates += 1
                if self.total_gradient_updates % target_update_interval == 0:
                    self.soft_update()

            # VALIDATION
            env_steps = step + 1
            if validate_fn is not None and env_steps % validate_every == 0:
                validate_fn(ts=env_steps)
