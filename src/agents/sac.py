from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Env

from src.utils.trajectories import collect_trajectories
from src.utils.torch import to_device


class SACReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, device: str | torch.device = "cpu"):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.size = size
        self.ptr = 0
        self.full = False
        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0
            self.full = True

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.action_buf[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.reward_buf[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device),
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
        env: Env,
        reward,
        policy,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        q_n_hidden_layers: int = 2,
        q_hidden: int = 256,
        lr: float = 1e-3,  # FIX: single lr for all (matching TTSA default)
        lr_actor: float = None,  # optional override; if None, uses lr
        lr_q: float = None,  # optional override; if None, uses lr
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 10_000,
        gamma: float = 0.99,
        polyak: float = 0.995,  # FIX: renamed from tau, default 0.995 (TTSA style)
        alpha: float = 0.2,
        autotune: bool = True,  # FIX: enabled by default (TTSA style)
        update_every: int = 50,  # FIX: added — how often to run updates
        update_num: int = 50,  # FIX: added — how many gradient steps per update trigger
        device: str | torch.device = "cpu",
    ):
        self.env = env
        self.reward = reward
        self.policy = policy
        self.obs = None

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.autotune = autotune
        self.update_every = update_every
        self.update_num = update_num

        self.device = torch.device(device)
        self.reward.to(self.device)
        self.policy.to(self.device)

        self.qf1 = SoftQNetwork(state_dim, action_dim, hidden=q_hidden, n_hidden_layers=q_n_hidden_layers).to(
            self.device
        )
        self.qf2 = SoftQNetwork(state_dim, action_dim, hidden=q_hidden, n_hidden_layers=q_n_hidden_layers).to(
            self.device
        )
        self.qf1_target = SoftQNetwork(state_dim, action_dim, hidden=q_hidden, n_hidden_layers=q_n_hidden_layers).to(
            self.device
        )
        self.qf2_target = SoftQNetwork(state_dim, action_dim, hidden=q_hidden, n_hidden_layers=q_n_hidden_layers).to(
            self.device
        )
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # FIX: freeze target network parameters — only updated via polyak averaging
        for p in self.qf1_target.parameters():
            p.requires_grad = False
        for p in self.qf2_target.parameters():
            p.requires_grad = False

        _lr_q = lr_q if lr_q is not None else lr
        _lr_actor = lr_actor if lr_actor is not None else lr

        # FIX: q_params saved as list so we can freeze/unfreeze during pi update
        self.q_params = list(self.qf1.parameters()) + list(self.qf2.parameters())
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=_lr_q)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=_lr_actor)

        if self.autotune:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=_lr_q)
            self.alpha = float(self.log_alpha.exp().item())

        self.rb = SACReplayBuffer(obs_dim=state_dim, action_dim=action_dim, size=buffer_size, device=self.device)
        self.global_step = 0

    def optimize(self, n_steps: int, inner_loss_fn=None, log_every: int = 0, n_log_traj: int = 3):
        if self.obs is None:
            self.obs, _ = self.env.reset()

        for _ in tqdm(range(n_steps), desc="SAC inner", leave=False):
            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.policy.sample_action(self.obs)
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            next_obs, env_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Reward from replay buffer is not used in _update:
            # we recompute learned reward on the sampled batch because reward network may change.
            self.rb.add(
                obs=self.obs,
                action=action,
                reward=0.0,
                next_obs=next_obs,
                done=float(terminated),
            )

            self.obs = next_obs if not done else None
            if self.obs is None:
                self.obs, _ = self.env.reset()

            # FIX: match TTSA update schedule — every `update_every` steps, do `update_num` gradient steps
            if (
                self.global_step >= self.learning_starts
                and len(self.rb) >= self.batch_size
                and self.global_step % self.update_every == 0
            ):
                for _ in range(self.update_num):
                    self._update()

            self.global_step += 1

            if inner_loss_fn is not None and log_every and self.global_step % log_every == 0:
                diag_trajs = collect_trajectories(
                    env=self.env, policy=self.policy, n=n_log_traj, max_steps=1000, desc="diag trajs", verbose=False
                )

                with torch.no_grad():
                    l_inner = inner_loss_fn(self.policy, self.reward, diag_trajs)

                    env_returns = []
                    learned_returns = []
                    logp_sums = []
                    reward_means = []
                    reward_stds = []
                    reward_mins = []
                    reward_maxs = []

                    for traj in diag_trajs:
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

                    self.obs, _ = self.env.reset()

    def _update(self):
        states, actions, _, next_states, dones = self.rb.sample(self.batch_size)

        # ── Critic update ──────────────────────────────────────────────────────
        with torch.no_grad():
            rewards = self.reward(states, actions)
            next_actions, next_log_pi, _ = self.policy.get_action(next_states)
            qf1_next = self.qf1_target(next_states, next_actions)
            qf2_next = self.qf2_target(next_states, next_actions)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi
            next_q_value = rewards + (1.0 - dones) * self.gamma * min_qf_next

        qf1_a = self.qf1(states, actions)
        qf2_a = self.qf2(states, actions)
        qf1_loss = F.mse_loss(qf1_a, next_q_value)
        qf2_loss = F.mse_loss(qf2_a, next_q_value)
        loss_q = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # ── Actor update ───────────────────────────────────────────────────────
        # FIX: freeze Q-params during actor update (saves compute, matches TTSA)
        for p in self.q_params:
            p.requires_grad = False

        pi_actions, log_pi, _ = self.policy.get_action(states)
        min_qf_pi = torch.min(self.qf1(states, pi_actions), self.qf2(states, pi_actions))
        actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        # ── Alpha (entropy coeff) update ───────────────────────────────────────
        if self.autotune:
            with torch.no_grad():
                _, log_pi_alpha, _ = self.policy.get_action(states)
            # FIX: correct autotune loss formula matching TTSA
            alpha_loss = -(self.log_alpha * (log_pi_alpha + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ── Target network update via polyak averaging ─────────────────────────
        # FIX: polyak averaging — slow EMA (polyak=0.995), NOT fast tau=0.005
        with torch.no_grad():
            for p, tp in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                tp.data.mul_(self.polyak)
                tp.data.add_((1.0 - self.polyak) * p.data)
            for p, tp in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                tp.data.mul_(self.polyak)
                tp.data.add_((1.0 - self.polyak) * p.data)
