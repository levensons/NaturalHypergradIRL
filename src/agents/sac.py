from tqdm import tqdm
import mlflow

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.env import Environment
from src.utils.policies import Policy
from src.utils.trajectories import collect_trajectories
from src.evaluation.metrics import inner_loss


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity

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
        self.state_buf[self.ptr] = torch.as_tensor(state, dtype=torch.float32).detach()
        self.action_buf[self.ptr] = torch.as_tensor(action, dtype=torch.float32).detach()
        self.reward_buf[self.ptr] = (torch.as_tensor(reward, dtype=torch.float32).detach().reshape(1))
        self.next_state_buf[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32).detach()
        self.done_buf[self.ptr] = (torch.as_tensor(done, dtype=torch.float32).detach().reshape(1))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = torch.randint(0, self.size, size=(batch_size,))

        states = self.state_buf[idxs]
        actions = self.action_buf[idxs]
        rewards = self.reward_buf[idxs].reshape(-1)
        next_states = self.next_state_buf[idxs]
        dones = self.done_buf[idxs].reshape(-1)

        return states, actions, rewards, next_states, dones


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

        self.q1 = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.q2 = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers)

        self.q1_target = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.q1_target.load_state_dict(self.q1.state_dict())

        self.q2_target = QFunction(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, capacity=replay_buffer_capacity)

        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

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
        eval_env: Environment = None,
        max_grad_norm: float = 1.0,
        critic_update_steps: int = 1,
        actor_update_steps: int = 1,
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-3
    ):
        state = train_env.reset()

        policy_params = list(self.policy.parameters())
        q1_params = list(self.q1.parameters())
        q2_params = list(self.q2.parameters())

        policy_optimizer = torch.optim.Adam(policy_params, lr=actor_lr)
        q1_optimizer = torch.optim.Adam(q1_params, lr=critic_lr)
        q2_optimizer = torch.optim.Adam(q2_params, lr=critic_lr)

        for ts in tqdm(range(total_steps), desc="SAC inner optimization", leave=False):
            if ts < learning_starts:
                action = train_env.get_random_action()
            else:
                with torch.no_grad():
                    action = self.policy.sample(state)

            next_state, reward, done = train_env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            if done:
                state = train_env.reset()
            else:
                state = next_state

            if len(self.replay_buffer) < learning_starts:
                continue

            # CRITIC UPDATE
            for _ in range(critic_update_steps):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

                with torch.no_grad():
                    next_actions, next_log_probs = self.policy.sample(next_states, return_log_probs=True)

                    next_q1 = self.q1_target(next_states, next_actions)
                    next_q2 = self.q2_target(next_states, next_actions)
                    next_q = torch.min(next_q1, next_q2)
                    target_q = rewards + self.gamma * (1.0 - dones) * (next_q - self.alpha * next_log_probs)

                current_q1 = self.q1(states, actions)
                current_q2 = self.q2(states, actions)

                q1_loss = F.mse_loss(current_q1, target_q)
                q2_loss = F.mse_loss(current_q2, target_q)

                q1_optimizer.zero_grad()
                torch.autograd.backward(q1_loss, inputs=q1_params)
                torch.nn.utils.clip_grad_norm_(q1_params, max_grad_norm)
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                torch.autograd.backward(q2_loss, inputs=q2_params)
                torch.nn.utils.clip_grad_norm_(q2_params, max_grad_norm)
                q2_optimizer.step()

            # ACTOR UPDATE
            for _ in range(actor_update_steps):
                states, _, _, _, _ = self.replay_buffer.sample(batch_size)

                new_actions, log_probs = self.policy.sample(
                    states, return_log_probs=True
                )

                q1 = self.q1(states, new_actions)
                q2 = self.q2(states, new_actions)
                q = torch.min(q1, q2)

                policy_loss = torch.mean(self.alpha * log_probs - q)

                policy_optimizer.zero_grad()
                torch.autograd.backward(policy_loss, inputs=policy_params)
                torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
                policy_optimizer.step()

            # CRITIC-TARGET UPDATE
            self.soft_update()

            if ts % 1000 == 0:
                self._validate_inner_loss(eval_env, n_eval_traj=10, max_steps=1000, step=ts)

    @torch.no_grad()
    def _validate_inner_loss(
        self,
        eval_env: Environment,
        n_eval_traj: int = 10,
        max_steps: int = 1000,
        step: int = 0,
    ):
        self.policy.eval()

        eval_trajs = collect_trajectories(
            env=eval_env,
            policy=self.policy,
            n=n_eval_traj,
            max_steps=max_steps,
            verbose=False,
        )

        l_inner = inner_loss(
            policy=self.policy,
            reward=eval_env.custom_reward,
            trajs=eval_trajs,
            discount=self.gamma,
            alpha=self.alpha,
        )

        mlflow.log_metric("sac/l_inner", float(l_inner), step=step)

        self.policy.train()

        return float(l_inner)
