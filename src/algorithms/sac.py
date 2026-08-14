from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.env import Environment
from src.utils.policies import Policy
from src.utils.torch import set_optimizer_lr


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity

        self.state_buf = torch.empty(capacity, state_dim, dtype=torch.float32)
        self.action_buf = torch.empty(capacity, action_dim, dtype=torch.float32)
        self.reward_buf = torch.empty(capacity, 1, dtype=torch.float32)
        self.next_state_buf = torch.empty(capacity, state_dim, dtype=torch.float32)
        self.terminated_buf = torch.empty(capacity, 1, dtype=torch.float32)

        self.ptr = 0
        self.size = 0

    def reset(self):
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, terminated):
        self.state_buf[self.ptr].copy_(state.detach())
        self.action_buf[self.ptr].copy_(action.detach())
        self.reward_buf[self.ptr, 0] = reward.detach()
        self.next_state_buf[self.ptr].copy_(next_state.detach())
        self.terminated_buf[self.ptr, 0] = terminated.detach()

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = torch.randint(0, self.size, size=(batch_size,))

        states = self.state_buf[idxs]
        actions = self.action_buf[idxs]
        rewards = self.reward_buf[idxs].reshape(-1)
        next_states = self.next_state_buf[idxs]
        terminateds = self.terminated_buf[idxs].reshape(-1)

        return states, actions, rewards, next_states, terminateds
    
    def extend_from_trajectories(self, trajs):
        for trajectory in tqdm(trajs, desc="Load trajectories into replay buffer", leave=False):
            states = trajectory["states"]
            actions = trajectory["actions"]

            if len(states) == 0:
                continue

            next_states = trajectory["next_states"]

            rewards = trajectory.get("rewards")
            if rewards is None:
                rewards = torch.zeros(len(states), dtype=states.dtype, device=states.device)

            terminateds = trajectory.get("terminateds")
            if terminateds is None:
                terminateds = torch.zeros(len(states),dtype=torch.bool, device=states.device)

            if not (len(states) == len(actions) == len(rewards) == len(next_states) == len(terminateds)):
                raise ValueError(
                    "Trajectory fields must have equal lengths: "
                    f"states={len(states)}, "
                    f"actions={len(actions)}, "
                    f"rewards={len(rewards)}, "
                    f"next_states={len(next_states)}, "
                    f"terminateds={len(terminateds)}"
                )

            for state, action, reward, next_state, terminated in zip(states, actions, rewards, next_states, terminateds):
                self.push(state, action, reward, next_state, terminated)

    def recalc_rewards(self, reward_fn, batch_size: int = 4096):
        if self.size == 0:
            return

        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)

            states = self.state_buf[start:end]
            actions = self.action_buf[start:end]

            rewards = reward_fn(states, actions)
            self.reward_buf[start:end, :] = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1)


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

    def reset_parameters(self):
        for layer in self.modules():
            if layer is not self and hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (B, state_dim)
        # actions: (B, action_dim)
        x = torch.cat([states, actions], dim=-1)
        out = self.backbone(x).squeeze(-1)
        return out  # (B,)


class SAC:
    def __init__(
        self,
        policy: Policy | nn.Module,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 1,
        gamma: float = 0.99,
        alpha: float = 1.0,
        tau: float = 0.005,
        replay_buffer_capacity: int = 1_000_000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer_capacity = replay_buffer_capacity

        self.policy = policy

        self.q1 = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.q2 = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers)

        self.q1_target = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.q1_target.load_state_dict(self.q1.state_dict())

        self.q2_target = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_capacity)

        self.policy_params = list(self.policy.parameters())
        self.critic_params = list(self.q1.parameters()) + list(self.q2.parameters())

        self.policy_optimizer = torch.optim.Adam(self.policy_params)
        self.critic_optimizer = torch.optim.Adam(self.critic_params)

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.global_gradient_update_step = 0

        self.q1.compile()
        self.q2.compile()
        self.q1_target.compile()
        self.q2_target.compile()

    def reset_optimizers(self):
        self.policy_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        self.policy_optimizer.state.clear()
        self.critic_optimizer.state.clear()
        self.global_gradient_update_step = 0

    def reset_policy(self):
        if not hasattr(self.policy, "reset_parameters"):
            raise TypeError("Policy must implement reset_parameters().")

        self.policy.reset_parameters()
        self.policy_optimizer.zero_grad()
        self.policy_optimizer.state.clear()

    def reset_critics(self):
        self.q1.reset_parameters()
        self.q2.reset_parameters()

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.critic_optimizer.zero_grad()
        self.critic_optimizer.state.clear()
        self.global_gradient_update_step = 0

    def reset_replay_buffer(self):
        self.replay_buffer.reset()

    def collect_random_rollout(self, env: Environment, n_steps: int):
        state = env.reset()

        for _ in tqdm(range(n_steps), desc="Collect random transitions", leave=False):
            action = env.get_random_action()

            next_state, reward, terminated, truncated = env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, terminated)

            if terminated.item() or truncated.item():
                state = env.reset()
            else:
                state = next_state

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
        batch_size: int = 256,
        max_grad_norm: float = 1.0,
        gradient_update_steps: int = 1,
        target_update_interval: int = 1,
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-3,
        validate_fn=None,
        validate_every: int = 1000,
    ):
        state = train_env.reset()

        set_optimizer_lr(self.policy_optimizer, actor_lr)
        set_optimizer_lr(self.critic_optimizer, critic_lr)

        for ts in tqdm(range(total_steps), desc="SAC inner optimization", leave=False):
            # COLLECTING
            with torch.no_grad():
                action = self.policy.sample(state)

            next_state, reward, terminated, truncated = train_env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, terminated)

            if terminated.item() or truncated.item():
                state = train_env.reset()
            else:
                state = next_state

            # UPDATE
            for _ in range(gradient_update_steps):
                # CRITIC
                states, actions, rewards, next_states, terminateds = self.replay_buffer.sample(batch_size)

                with torch.no_grad():
                    next_actions, next_log_probs = self.policy.sample(next_states, return_log_probs=True)

                    next_q1 = self.q1_target(next_states, next_actions)
                    next_q2 = self.q2_target(next_states, next_actions)
                    next_q = torch.min(next_q1, next_q2)
                    target_q = rewards + self.gamma * (1.0 - terminateds) * (next_q - self.alpha * next_log_probs)

                current_q1 = self.q1(states, actions)
                current_q2 = self.q2(states, actions)

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

                q1 = self.q1(states, new_actions)
                q2 = self.q2(states, new_actions)
                q = torch.min(q1, q2)

                policy_loss = torch.mean(self.alpha * log_probs - q)

                self.policy_optimizer.zero_grad()
                torch.autograd.backward(policy_loss, inputs=self.policy_params)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy_params, max_grad_norm)
                self.policy_optimizer.step()

                # CRITIC-TARGET SOFT-UPDATE
                self.global_gradient_update_step += 1
                if self.global_gradient_update_step % target_update_interval == 0:
                    self.soft_update()

            # VALIDATION
            if validate_fn is not None and ts % validate_every == 0:
                validate_fn(ts)
