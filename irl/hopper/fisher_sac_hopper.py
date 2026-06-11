import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import SAC
from tqdm import tqdm


def flat_grad(params):
    return torch.cat([p.flatten() for p in params])


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def assign_flat_gradients(module, flat_grad: torch.Tensor):
    i = 0
    for p in module.parameters():
        n = p.numel()
        grad_chunk = flat_grad[i : i + n]

        if grad_chunk.numel() != n:
            raise ValueError(
                "Flat gradient has incorrect size: not enough elements for model parameters."
            )

        p.grad = grad_chunk.reshape(p.shape).clone()
        i += n

    if i != flat_grad.numel():
        raise ValueError(
            "Flat gradient has incorrect size: too many elements for model parameters."
        )


def get_env_dims(env: Env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return state_dim, action_dim

class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([states, actions], dim=-1)
        return self.net(sa).squeeze(-1)

    def discounted_rewards(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        rewards = self.forward(states, actions)
        ts = torch.arange(
            states.size(0),
            dtype=torch.float32,
            device=states.device,
        )
        discounts = torch.pow(self.gamma.to(states.device), ts)
        return discounts * rewards

    def trajectory_return(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()


class SACReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        size: int,
        device: torch.device,
    ):
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
        max_idx = len(self)
        idxs = np.random.randint(0, max_idx, size=batch_size)

        obs = torch.tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.action_buf[idxs], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.reward_buf[idxs], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device)

        return obs, actions, rewards, next_obs, dones


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        hidden: int = 128,
        log_std_max=2,
        log_std_min=-5
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def forward(self, states: torch.Tensor):
        h = self.net(states)

        mean = self.mean_head(h)
        log_std = self.log_std_head(h)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)

        return mean, log_std

    def get_action(self, states: torch.Tensor):
        mean, log_std = self.forward(states)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(
            self.action_scale * (1.0 - y_t.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action

    @staticmethod
    def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def action_to_normalized(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_bias) / self.action_scale

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        mean, log_std = self.forward(states)
        std = log_std.exp()

        normalized_action = self.action_to_normalized(actions)
        normalized_action = torch.clamp(normalized_action, -1.0 + 1e-6, 1.0 - 1e-6)

        pre_tanh_action = self.atanh(normalized_action)

        normal = torch.distributions.Normal(mean, std)

        log_prob = normal.log_prob(pre_tanh_action)

        correction = torch.log(
            self.action_scale * (1.0 - normalized_action.pow(2)) + 1e-6
        )

        log_prob = log_prob - correction
        log_prob = log_prob.sum(dim=-1)

        return log_prob

    def sample_action(self, state, deterministic: bool = False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.get_action(state_tensor)
            else:
                action, _, _ = self.get_action(state_tensor)

        return action.squeeze(0).cpu().numpy()


class SACInnerOptimizer:
    def __init__(
        self,
        env: Env,
        reward: Reward,
        policy: Policy,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        lr_actor: float = 3e-4,
        lr_q: float = 1e-3,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 5_000,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        autotune: bool = True,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
        device: str = "cpu",
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
        self.tau = tau
        self.alpha = alpha
        self.autotune = autotune
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency

        self.device = torch.device(device)

        self.reward.to(self.device)
        self.policy.to(self.device)

        self.qf1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.qf2 = SoftQNetwork(state_dim, action_dim).to(self.device)

        self.qf1_target = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.qf2_target = SoftQNetwork(state_dim, action_dim).to(self.device)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=lr_q,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr_actor,
        )

        if self.autotune:
            self.target_entropy = -float(action_dim)

            self.log_alpha = torch.zeros(
                1,
                requires_grad=True,
                device=self.device,
            )

            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=lr_q,
            )

            self.alpha = self.log_alpha.exp().item()

        self.rb = SACReplayBuffer(
            obs_dim=state_dim,
            action_dim=action_dim,
            size=buffer_size,
            device=self.device,
        )

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
                action = np.clip(
                    action,
                    self.env.action_space.low,
                    self.env.action_space.high,
                )

            next_obs, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            state_t = torch.tensor(
                self.obs,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            action_t = torch.tensor(
                action,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            learned_r = self.learned_reward(state_t, action_t).item()

            self.rb.add(
                obs=self.obs,
                action=action,
                reward=learned_r,
                next_obs=next_obs,
                done=float(done),
            )

            if done:
                self.obs, _ = self.env.reset()
            else:
                self.obs = next_obs

            if self.global_step > self.learning_starts and len(self.rb) >= self.batch_size:
                self.update()

            self.global_step += 1

    def update(self):
        states, actions, old_rewards, next_states, dones = self.rb.sample(self.batch_size)

        with torch.no_grad():
            rewards = self.reward(states, actions)

            next_actions, next_log_pi, _ = self.policy.get_action(next_states)

            qf1_next = self.qf1_target(next_states, next_actions)
            qf2_next = self.qf2_target(next_states, next_actions)

            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi

            next_q_value = rewards + (1.0 - dones) * self.gamma * min_qf_next

        qf1_values = self.qf1(states, actions)
        qf2_values = self.qf2(states, actions)

        qf1_loss = F.mse_loss(qf1_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if self.global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi_actions, log_pi, _ = self.policy.get_action(states)

                qf1_pi = self.qf1(states, pi_actions)
                qf2_pi = self.qf2(states, pi_actions)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi_alpha, _ = self.policy.get_action(states)

                    alpha_loss = (
                        -self.log_alpha.exp()
                        * (log_pi_alpha + self.target_entropy)
                    ).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()

                    self.alpha = self.log_alpha.exp().item()

        if self.global_step % self.target_network_frequency == 0:
            with torch.no_grad():
                for param, target_param in zip(
                    self.qf1.parameters(),
                    self.qf1_target.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.qf2.parameters(),
                    self.qf2_target.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )

def collect_trajectories(env: Env, policy: Policy, n: int, max_steps=1000):
    trajs = []

    for _ in tqdm(range(n), desc="collect trajs", leave=False):
        states, actions, env_rewards = [], [], []
        s, _ = env.reset()

        for _ in range(max_steps):
            a = policy.sample_action(s)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            s_next, r, terminated, truncated, _ = env.step(a)

            states.append(torch.tensor(s, dtype=torch.float32))
            actions.append(torch.tensor(a, dtype=torch.float32))
            env_rewards.append(r)
            s = s_next

            if terminated or truncated:
                break

        traj = {
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "env_rewards": env_rewards,
        }
        trajs.append(traj)

    return trajs


class OuterOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        max_grad_norm: float = None,
        fisher_reg: float = 1e-2,
        gamma: float = 0.99,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.gamma = gamma

        self.max_grad_norm = max_grad_norm
        self.raw_grad_norm = None
        self.clipped_grad_norm = None

        self.optimizer = torch.optim.SGD(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)

    def update_policy(self, policy: Policy):
        self.policy = policy

    def score(self, states, actions) -> torch.Tensor:
        # log_prob_mean = self.policy.log_prob(states, actions).mean()
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def discounted_score(self, states, actions) -> torch.Tensor:
        ts = torch.arange(
            states.size(0),
            dtype=torch.float32,
            device=states.device,
        )
        discounts = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float32, device=states.device),
            ts,
        )
        log_prob_sum = (discounts * self.policy.log_prob(states, actions)).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def fisher(self, trajs) -> torch.Tensor:
        d = num_params(self.policy)
        F = torch.zeros(d, d)

        for traj in trajs:
            states, actions = traj["states"], traj["actions"]
            s = self.score(states, actions)
            F += torch.outer(s, s)

        F /= len(trajs)
        F += self.fisher_reg * torch.eye(d)
        return F

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        d = num_params(self.policy)
        grad = torch.zeros(d)

        for traj in expert_trajs:
            states, actions = traj["states"], traj["actions"]
            # grad += self.score(states, actions)
            grad += self.discounted_score(states, actions)

        return -grad / len(expert_trajs)

    def cross_derivative(self, trajs) -> torch.Tensor:
        d_theta = num_params(self.policy)
        d_phi = num_params(self.reward)
        H = torch.zeros(d_theta, d_phi)

        for traj in trajs:
            states, actions = traj["states"], traj["actions"]
            s_theta = self.score(states, actions)

            reward_sum = self.reward.trajectory_return(states, actions)
            grads_phi = torch.autograd.grad(reward_sum, self.reward.parameters())
            g_phi = flat_grad(grads_phi)

            H += torch.outer(s_theta, g_phi)

        return -H / len(trajs)

    def cross_derivative_vec_product(self, trajs, v: torch.Tensor) -> torch.Tensor:
        d_phi = num_params(self.reward)
        result = torch.zeros(d_phi)

        for traj in trajs:
            states, actions = traj["states"], traj["actions"]
            s_theta = self.score(states, actions)

            reward_sum = self.reward.trajectory_return(states, actions)
            grads_phi = torch.autograd.grad(reward_sum, self.reward.parameters())
            g_phi = flat_grad(grads_phi)

            result += torch.dot(s_theta, v) * g_phi

        return -result / len(trajs)

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)
        outer_grad = self.outer_grad(expert_trajs)

        fisher_inv_outer_grad = torch.linalg.solve(fisher, outer_grad)
        hypergrad = self.cross_derivative_vec_product(
            agent_trajs, fisher_inv_outer_grad
        )

        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(fisher)
            min_eig = eigvals.min().item()
            max_eig = eigvals.max().item()

            cond_number = max_eig / max(min_eig, 1e-12)
            print(
                f"Fisher stats | "
                f"min_eig={min_eig:.3e} | "
                f"max_eig={max_eig:.3e} | "
                f"cond={cond_number:.3e} | "
                f"outer_grad_norm={outer_grad.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e}"
            )

        return hypergrad

    def step(self, expert_trajs, agent_trajs) -> torch.Tensor:
        hypergrad = self.hypergradient(expert_trajs, agent_trajs)

        self.raw_grad_norm = hypergrad.norm().item()
        if self.max_grad_norm is not None and self.raw_grad_norm > self.max_grad_norm:
            hypergrad = hypergrad * (self.max_grad_norm / self.raw_grad_norm)

        self.clipped_grad_norm = hypergrad.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.reward, hypergrad)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return hypergrad


class InnerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, policy: Policy, reward: Reward, trajs) -> torch.Tensor:
        losses = []

        for traj in trajs:
            states = traj["states"]
            actions = traj["actions"]

            log_probs = policy.log_prob(states, actions)
            discounted_rewards = reward.discounted_rewards(states, actions)

            losses.append((log_probs - discounted_rewards).sum())

        return torch.stack(losses).mean()


class OuterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, policy: Policy, expert_trajs):
        losses = []

        for traj in expert_trajs:
            states = traj["states"]
            actions = traj["actions"]

            log_probs = policy.log_prob(states, actions)
            losses.append(-log_probs.sum())

        return torch.stack(losses).mean()


class RankCorr(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def rankdata(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float()
        sorted_x, order = torch.sort(x)
        ranks = torch.empty_like(x, dtype=torch.float32)

        n = x.numel()
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            avg_rank = 0.5 * (i + j)
            ranks[order[i : j + 1]] = avg_rank
            i = j + 1

        return ranks

    @staticmethod
    def pearson_corr(
        x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        x = x.float()
        y = y.float()
        x = x - x.mean()
        y = y - y.mean()

        denom = torch.sqrt(torch.sum(x**2) * torch.sum(y**2))
        if denom < eps:
            return torch.tensor(torch.nan)

        return torch.sum(x * y) / denom

    @torch.no_grad()
    def forward(self, reward: Reward, trajs):
        reward.eval()

        env_returns, learned_returns = [], []
        for traj in trajs:
            env_return = torch.tensor(traj["env_rewards"], dtype=torch.float32).sum()
            env_returns.append(env_return)

            learned_return = reward.trajectory_return(traj["states"], traj["actions"])
            learned_returns.append(learned_return)

        env_returns = torch.stack(env_returns)
        learned_returns = torch.stack(learned_returns)

        env_ranks = self.rankdata(env_returns)
        learned_ranks = self.rankdata(learned_returns)
        return self.pearson_corr(env_ranks, learned_ranks).item()


class PolicyNLL(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, policy: Policy, expert_trajs) -> float:
        policy.eval()

        nll = 0.0
        for traj in expert_trajs:
            log_probs = policy.log_prob(traj["states"], traj["actions"])
            nll += -log_probs.sum().item()

        return nll / len(expert_trajs)


class SB3PolicyWrapper:
        def __init__(self, model, action_space):
            self.model = model
            self.action_low = action_space.low
            self.action_high = action_space.high

        def sample_action(self, state):
            action, _ = self.model.predict(state, deterministic=True)
            return np.clip(action, self.action_low, self.action_high)

        def eval(self):
            pass


def train_expert(env_name, save_path, total_timesteps, seed, verbose=1):
    sb3_env = gym.make(env_name)
    sb3_env.reset(seed=seed)
    sb3_env.action_space.seed(seed)
    sb3_env.observation_space.seed(seed)

    model = SAC(
        "MlpPolicy",
        sb3_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=verbose,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(save_path)
    print(f"Saved SAC expert to {save_path}.zip")

    expert_policy = SB3PolicyWrapper(model, sb3_env.action_space)
    sb3_env.close()
    return expert_policy


def load_expert(env_name, load_path, seed):
    tmp_env = gym.make(env_name)
    tmp_env.reset(seed=seed)
    tmp_env.action_space.seed(seed)
    tmp_env.observation_space.seed(seed)

    model = SAC.load(load_path)

    expert_policy = SB3PolicyWrapper(model, tmp_env.action_space)
    tmp_env.close()

    return expert_policy


def save_irl_checkpoint(
    path: str,
    policy: Policy,
    reward: Reward,
    state_dim: int,
    action_dim: int,
):
    checkpoint = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "policy_state_dict": policy.state_dict(),
        "reward_state_dict": reward.state_dict(),
    }

    torch.save(checkpoint, path)
    print(f"Saved IRL checkpoint to {path}")


def load_irl_checkpoint(path: str, action_low, action_high):
    checkpoint = torch.load(path, map_location="cpu")

    policy = Policy(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        action_low=action_low,
        action_high=action_high,
        hidden=64,
    )

    reward = Reward(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
    )

    policy.load_state_dict(checkpoint["policy_state_dict"])
    reward.load_state_dict(checkpoint["reward_state_dict"])

    return policy, reward, checkpoint


def bilevel_irl(
    n_outer_steps: int,
    n_inner_steps: int,
    n_agent_traj: int,
    n_expert_traj: int,
    n_eval_traj: int,
    lr_outer: float,
    lr_inner: float,
    seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("Hopper-v5")
    env.reset(seed=seed)

    state_dim, action_dim = get_env_dims(env)

    print("=" * 50)
    print("Step 1: loading/training an expert")
    print("=" * 50)

    expert_path = "sac_hopper_expert"

    if os.path.exists(expert_path + ".zip"):
        print(f"Loading saved SAC expert from {expert_path}.zip")
        expert_policy = load_expert(
            env_name="Hopper-v5",
            load_path=expert_path,
            seed=seed,
        )

    else:
        print("Saved SAC expert not found. Training a new one.")
        expert_policy = train_expert(
            env_name="Hopper-v5",
            save_path=expert_path,
            total_timesteps=500_000,
            seed=seed,
            verbose=1,
        )

    print("\nStep 2: expert's trajectories")
    expert_trajs = collect_trajectories(env, expert_policy, n=n_expert_traj)

    avg_len = np.mean([len(t["states"]) for t in expert_trajs])
    avg_return = np.mean([sum(t["env_rewards"]) for t in expert_trajs])
    print(
        f"Collected {n_expert_traj} trajectories | avg len = {avg_len:.1f} | avg return = {avg_return:.1f}"
    )

    reward = Reward(state_dim, action_dim)
    
    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=lr_outer,
        fisher_reg=1.0,
        max_grad_norm=None,
        gamma=0.99,
    )

    sac_env = gym.make("Hopper-v5")
    sac_env.reset(seed=seed + 1)
    inner_optimizer = SACInnerOptimizer(
        env=sac_env,
        reward=reward,
        policy=policy,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        lr_actor=lr_inner,
        lr_q=1e-3,
        buffer_size=300_000,
        batch_size=256,
        learning_starts=500,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        autotune=True,
        policy_frequency=2,
        target_network_frequency=1,
    )

    outer_loss_fn = OuterLoss()
    rank_corr_fn = RankCorr()
    policy_nll_fn = PolicyNLL()

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

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        "fisher_sac_hopper_checkpoint.pt",
    )

    best_l_outer = float("inf")

    print("\nStep 3: bilevel optimization")
    print("=" * 190)
    print(
        f"{'Step':>5} | "
        f"{'L_outer':>10} | "
        f"{'agent_len':>10} | "
        f"{'expert_len':>10} | "
        f"{'agent_ret':>10} | "
        f"{'expert_ret':>10} | "
        f"{'RankCorr':>9} | "
        f"{'PolicyNLL':>10} | "
        f"{'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | "
        f"{'lr_outer':>12}"
    )
    print("-" * 190)

    for outer_step in range(1, n_outer_steps + 1):
        inner_optimizer.optimize(n_inner_steps)
        agent_trajs = collect_trajectories(env, policy, n_agent_traj)
        outer_optimizer.step(expert_trajs, agent_trajs)

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss_fn(policy, expert_trajs).item()

        agent_len = np.mean([len(t["states"]) for t in agent_trajs])
        expert_len = np.mean([len(t["states"]) for t in expert_trajs])

        agent_return = np.mean([sum(t["env_rewards"]) for t in agent_trajs])
        expert_return = np.mean([sum(t["env_rewards"]) for t in expert_trajs])

        rank_corr = rank_corr_fn(reward, expert_trajs + agent_trajs)
        policy_nll = policy_nll_fn(policy, expert_trajs)

        history["l_outer"].append(l_outer)
        history["agent_len"].append(agent_len)
        history["expert_len"].append(expert_len)
        history["agent_return"].append(agent_return)
        history["expert_return"].append(expert_return)
        history["rank_corr"].append(rank_corr)
        history["policy_nll"].append(policy_nll)
        history["raw_hypgrad_norm"].append(raw_hypgrad_norm)
        history["clipped_hypgrad_norm"].append(clipped_hypgrad_norm)
        history["lr_outer"].append(lr_outer_current)

        if l_outer < best_l_outer:
            best_l_outer = l_outer

            save_irl_checkpoint(
                path=best_checkpoint_path,
                policy=policy,
                reward=reward,
                state_dim=state_dim,
                action_dim=action_dim,
            )

        print(
            f"{outer_step:>5} | "
            f"{l_outer:>10.3f} | "
            f"{agent_len:>10.1f} | "
            f"{expert_len:>10.1f} | "
            f"{agent_return:>10.1f} | "
            f"{expert_return:>10.1f} | "
            f"{rank_corr:>9.3f} | "
            f"{policy_nll:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | "
            f"{clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )

    print("\nStep 4: final evaluation")
    print("=" * 60)

    if os.path.exists(best_checkpoint_path):
        policy, reward, checkpoint = load_irl_checkpoint(
            best_checkpoint_path,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        )
        print(f"Loaded best checkpoint from {best_checkpoint_path}")

    policy.eval()
    reward.eval()

    expert_test_trajs = collect_trajectories(
        env, expert_policy, n=n_eval_traj, max_steps=1000
    )
    agent_test_trajs = collect_trajectories(env, policy, n=n_eval_traj, max_steps=1000)
    random_test_trajs = collect_trajectories(
        env,
        Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden=64,
        ),
        n=n_eval_traj,
        max_steps=1000,
    )

    rank_pool = expert_test_trajs + agent_test_trajs + random_test_trajs

    policy_nll = PolicyNLL()(policy, expert_test_trajs)
    rank_corr = RankCorr()(reward, rank_pool)

    env_reward = float(np.mean([sum(t["env_rewards"]) for t in agent_test_trajs]))
    expert_env_reward = float(
        np.mean([sum(t["env_rewards"]) for t in expert_test_trajs])
    )
    random_env_reward = float(
        np.mean([sum(t["env_rewards"]) for t in random_test_trajs])
    )

    metrics = {
        "policy_nll": policy_nll,
        "env_reward": env_reward,
        "expert_env_reward": expert_env_reward,
        "random_env_reward": random_env_reward,
        "rank_corr": rank_corr,
    }

    print(f"PolicyNLL  = {policy_nll:.4f}")
    print(f"EnvReward  = {env_reward:.1f}")
    print(f"Expert     = {expert_env_reward:.1f}")
    print(f"Random     = {random_env_reward:.1f}")
    print(f"RankCorr   = {rank_corr:.4f}")

    sac_env.close()
    env.close()

    return policy, reward, history, metrics


if __name__ == "__main__":
    policy, reward, history, metrics = bilevel_irl(
        n_outer_steps=100,
        n_inner_steps=100000,
        n_agent_traj=1000,
        n_expert_traj=1000,
        n_eval_traj=1000,
        lr_outer=1e-6,
        lr_inner=1e-4,
        seed=42,
    )
