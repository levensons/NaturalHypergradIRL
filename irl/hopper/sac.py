"""
SAC-компоненты внутреннего агента: SACReplayBuffer, SoftQNetwork, SACInnerOptimizer.
Выделены из fisher_sac_hopper.py без изменения алгоритма.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SACReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, device: torch.device):
        self.obs_buf      = np.zeros((size, obs_dim),    dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim),    dtype=np.float32)
        self.action_buf   = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_buf   = np.zeros((size,),            dtype=np.float32)
        self.done_buf     = np.zeros((size,),            dtype=np.float32)
        self.size = size
        self.ptr  = 0
        self.full = False
        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr]      = obs
        self.action_buf[self.ptr]   = action
        self.reward_buf[self.ptr]   = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr]     = done
        self.ptr += 1
        if self.ptr == self.size:
            self.ptr  = 0
            self.full = True

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.tensor(self.obs_buf[idxs],      dtype=torch.float32, device=self.device),
            torch.tensor(self.action_buf[idxs],   dtype=torch.float32, device=self.device),
            torch.tensor(self.reward_buf[idxs],   dtype=torch.float32, device=self.device),
            torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.done_buf[idxs],     dtype=torch.float32, device=self.device),
        )


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1)).squeeze(-1)


class SACInnerOptimizer:
    """SAC-оптимизатор внутренней задачи. Параметры берутся из config.inner_agent.sac."""

    def __init__(
        self,
        env,
        reward,
        policy,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        lr_actor: float = 3e-4,
        lr_q: float = 1e-3,
        buffer_size: int = 300_000,
        batch_size: int = 256,
        learning_starts: int = 500,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        autotune: bool = True,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
        device: str = "cpu",
    ):
        self.env    = env
        self.reward = reward
        self.policy = policy
        self.obs    = None

        self.state_dim               = state_dim
        self.action_dim              = action_dim
        self.batch_size              = batch_size
        self.learning_starts         = learning_starts
        self.gamma                   = gamma
        self.tau                     = tau
        self.alpha                   = alpha
        self.autotune                = autotune
        self.policy_frequency        = policy_frequency
        self.target_network_frequency = target_network_frequency

        self.device = torch.device(device)
        self.reward.to(self.device)
        self.policy.to(self.device)

        self.qf1        = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.qf2        = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.qf1_target = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.qf2_target = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr_q)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        if self.autotune:
            self.target_entropy = -float(action_dim)
            self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_q)
            self.alpha = self.log_alpha.exp().item()

        self.rb = SACReplayBuffer(
            obs_dim=state_dim, action_dim=action_dim, size=buffer_size, device=self.device)
        self.global_step = 0

    def learned_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.reward(states, actions)

    def optimize(self, n_steps: int):
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

            state_t  = torch.tensor(self.obs,   dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            learned_r = self.learned_reward(state_t, action_t).item()

            self.rb.add(obs=self.obs, action=action, reward=learned_r,
                        next_obs=next_obs, done=float(done))

            self.obs = next_obs if not done else None
            if self.obs is None:
                self.obs, _ = self.env.reset()

            if self.global_step > self.learning_starts and len(self.rb) >= self.batch_size:
                self._update()

            self.global_step += 1

    def _update(self):
        states, actions, _, next_states, dones = self.rb.sample(self.batch_size)

        with torch.no_grad():
            rewards = self.reward(states, actions)
            next_actions, next_log_pi, _ = self.policy.get_action(next_states)
            qf1_next = self.qf1_target(next_states, next_actions)
            qf2_next = self.qf2_target(next_states, next_actions)
            min_qf_next  = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi
            next_q_value = rewards + (1.0 - dones) * self.gamma * min_qf_next

        qf1_loss = F.mse_loss(self.qf1(states, actions), next_q_value)
        qf2_loss = F.mse_loss(self.qf2(states, actions), next_q_value)
        self.q_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.q_optimizer.step()

        if self.global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi_actions, log_pi, _ = self.policy.get_action(states)
                min_qf_pi  = torch.min(self.qf1(states, pi_actions), self.qf2(states, pi_actions))
                actor_loss = (self.alpha * log_pi - min_qf_pi).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi_alpha, _ = self.policy.get_action(states)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi_alpha + self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        if self.global_step % self.target_network_frequency == 0:
            with torch.no_grad():
                for p, tp in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
                for p, tp in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
