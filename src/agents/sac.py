from tqdm import tqdm
import mlflow

from gymnasium import Env
import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.seeding import set_env_seed
from src.utils.trajectories import collect_trajectories
from src.utils.torch import to_device
from src.evaluation.metrics import inner_loss


class SACReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        size: int,
        device: str | torch.device = "cpu",
    ):
        self.max_size = int(size)
        self.device = torch.device(device)

        self.state_buf = torch.zeros(
            self.max_size,
            state_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.next_state_buf = torch.zeros(
            self.max_size,
            state_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_buf = torch.zeros(
            self.max_size,
            action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.reward_buf = torch.zeros(
            self.max_size,
            dtype=torch.float32,
            device=self.device,
        )
        self.terminal_buf = torch.zeros(
            self.max_size,
            dtype=torch.float32,
            device=self.device,
        )

        self.ptr = 0
        self.full = False

    def add(self, state, action, reward: float, next_state, terminal: bool | float):
        self.state_buf[self.ptr] = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_buf[self.ptr] = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=self.device,
        )
        self.reward_buf[self.ptr] = torch.as_tensor(
            reward,
            dtype=torch.float32,
            device=self.device,
        )
        self.next_state_buf[self.ptr] = torch.as_tensor(
            next_state,
            dtype=torch.float32,
            device=self.device,
        )
        self.terminal_buf[self.ptr] = torch.as_tensor(
            terminal,
            dtype=torch.float32,
            device=self.device,
        )

        self.ptr += 1
        if self.ptr == self.max_size:
            self.ptr = 0
            self.full = True

    def __len__(self):
        return self.max_size if self.full else self.ptr

    def sample(self, batch_size: int):
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indices = torch.randint(0, len(self), size=(batch_size,), device=self.device)

        return (
            self.state_buf[indices],
            self.action_buf[indices],
            self.reward_buf[indices],
            self.next_state_buf[indices],
            self.terminal_buf[indices],
        )


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        n_hidden_layers: int = 2,
    ):
        super().__init__()

        layers = [nn.Linear(state_dim + action_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1)).squeeze(-1)


class SACInnerOptimizer:
    def __init__(
        self,
        env_id: str,
        train_env_seed: int,
        eval_env_seed: int,
        reward,
        policy,
        state_dim: int,
        action_dim: int,
        q_n_hidden_layers: int = 2,
        q_hidden: int = 256,
        lr: float = 1e-3,
        lr_actor: float | None = None,
        lr_q: float | None = None,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 10_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        alpha: float = 1.0,
        update_every: int = 50,
        update_num: int = 50,
        scheduler_gamma: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        self.env_id = env_id
        self.train_env_seed = train_env_seed
        self.eval_env_seed = eval_env_seed

        self.train_env = gym.make(env_id)
        self.eval_env = gym.make(env_id)

        set_env_seed(self.train_env, train_env_seed)
        set_env_seed(self.eval_env, eval_env_seed)

        self.reward = reward
        self.policy = policy
        self.state = None

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        self.gamma = gamma
        self.alpha = alpha

        self.polyak = polyak
        self.update_every = update_every
        self.update_num = update_num
        self.scheduler_gamma = scheduler_gamma

        self.device = torch.device(device)
        self.reward.to(self.device)
        self.policy.to(self.device)

        self.qf1 = SoftQNetwork(state_dim, action_dim, q_hidden, q_n_hidden_layers).to(self.device)
        self.qf2 = SoftQNetwork(state_dim, action_dim, q_hidden, q_n_hidden_layers).to(self.device)

        self.qf1_target = SoftQNetwork(state_dim, action_dim, q_hidden, q_n_hidden_layers).to(self.device)
        self.qf2_target = SoftQNetwork(state_dim, action_dim, q_hidden, q_n_hidden_layers).to(self.device)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        for p in self.qf1_target.parameters():
            p.requires_grad = False
        for p in self.qf2_target.parameters():
            p.requires_grad = False

        lr_q_final = lr if lr_q is None else lr_q
        lr_actor_final = lr if lr_actor is None else lr_actor

        self.q_params = list(self.qf1.parameters()) + list(self.qf2.parameters())
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=lr_q_final)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor_final)

        self.q_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.q_optimizer, gamma=scheduler_gamma)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=scheduler_gamma)

        self.rb = SACReplayBuffer(state_dim, action_dim, buffer_size, self.device)

        self.global_step = 0

        self._closed = False

    @torch.no_grad()
    def _validate(self, n_eval_traj):
        self.policy.eval()
        self.reward.eval()

        set_env_seed(self.eval_env, self.eval_env_seed)
        eval_trajs = collect_trajectories(self.eval_env, self.policy, n_eval_traj, max_steps=1000, verbose=False)

        l_inner = inner_loss(self.policy, self.reward, eval_trajs, self.gamma, self.alpha)

        env_returns = []
        learned_returns = []
        logp_sums = []
        reward_means = []
        reward_stds = []
        reward_mins = []
        reward_maxs = []

        for traj in eval_trajs:
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)
            T = states.size(0)

            rewards = self.reward(states, actions)
            log_probs = self.policy.log_prob(states, actions)

            ts = torch.arange(T, dtype=torch.float32, device=self.device)
            discounts = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), ts)

            learned_return = (discounts * rewards).sum()

            env_returns.append(float(sum(traj["env_rewards"])))
            learned_returns.append(learned_return.item())
            logp_sums.append(log_probs.sum().item())
            reward_means.append(rewards.mean().item())
            reward_stds.append(rewards.std().item())
            reward_mins.append(rewards.min().item())
            reward_maxs.append(rewards.max().item())

        env_ret = float(np.mean(env_returns))
        learned_ret = float(np.mean(learned_returns))
        logp_sum = float(np.mean(logp_sums))
        r_mean = float(np.mean(reward_means))
        r_std = float(np.mean(reward_stds))
        r_min = float(np.mean(reward_mins))
        r_max = float(np.mean(reward_maxs))

        tqdm.write(
            f"   [inner] gstep={self.global_step} "
            f"L_inner={l_inner:.1f} "
            f"env_ret={env_ret:.1f} "
            f"learned_ret={learned_ret:.1f} "
            f"logp_sum={logp_sum:.1f} "
            f"r_mean={r_mean:.3f} "
            f"r_std={r_std:.3f} "
            f"r_min={r_min:.3f} "
            f"r_max={r_max:.3f}"
        )

        mlflow.log_metrics(
            {
                "inner_loss": l_inner,
                "inner_env_return": env_ret,
                "inner_learned_return": learned_ret,
                "reward_mean": r_mean,
                "reward_std": r_std,
                "reward_min": r_min,
                "reward_max": r_max,
            },
            step=self.global_step,
        )

    def optimize(self, n_steps: int, eval_every: int = 0, n_eval_traj: int = 10):
        set_env_seed(self.train_env, self.train_env_seed)

        if self.state is None:
            self.state, _ = self.train_env.reset()

        for _ in tqdm(range(n_steps), desc="SAC inner", leave=False):
            self.policy.train()
            self.reward.train()

            if self.global_step < self.learning_starts:
                action = self.train_env.action_space.sample()
            else:
                action = self.policy.sample_action(self.state)

            next_state, _, terminated, truncated, _ = self.train_env.step(action)

            self.rb.add(
                state=self.state,
                action=action,
                reward=0.0,
                next_state=next_state,
                terminal=terminated,
            )

            if not (terminated or truncated):
                self.state = next_state
            else:
                self.state, _ = self.train_env.reset()

            if (
                self.global_step >= self.learning_starts
                and len(self.rb) >= self.batch_size
                and self.global_step % self.update_every == 0
            ):
                for _ in range(self.update_num):
                    self._update()

                self.q_scheduler.step()
                self.actor_scheduler.step()

            self.global_step += 1

            # VALIDATION
            if eval_every and self.global_step % eval_every == 0:
                self._validate(n_eval_traj)

    def _update(self):
        states, actions, _, next_states, terminals = self.rb.sample(self.batch_size)

        with torch.no_grad():
            rewards = self.reward(states, actions)
            next_actions, next_log_pi, _ = self.policy.get_action(next_states)

            qf1_next = self.qf1_target(next_states, next_actions)
            qf2_next = self.qf2_target(next_states, next_actions)

            soft_next_value = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi
            next_q_value = rewards + (1.0 - terminals) * self.gamma * soft_next_value

        qf1_pred = self.qf1(states, actions)
        qf2_pred = self.qf2(states, actions)

        qf1_loss = F.mse_loss(qf1_pred, next_q_value)
        qf2_loss = F.mse_loss(qf2_pred, next_q_value)
        q_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ── Actor update ───────────────────────────────────────────────────────
        for p in self.q_params:
            p.requires_grad = False

        pi_actions, log_pi, _ = self.policy.get_action(states)
        qf1_pi = self.qf1(states, pi_actions)
        qf2_pi = self.qf2(states, pi_actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        # ── Target network update ──────────────────────────────────────────────
        with torch.no_grad():
            for p, tp in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                tp.data.mul_(self.polyak)
                tp.data.add_((1.0 - self.polyak) * p.data)

            for p, tp in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                tp.data.mul_(self.polyak)
                tp.data.add_((1.0 - self.polyak) * p.data)


    def close(self) -> None:
        if self._closed:
            return

        self.train_env.close()
        self.eval_env.close()
        self._closed = True


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
