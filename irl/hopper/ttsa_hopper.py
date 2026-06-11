import os
os.environ.setdefault('MUJOCO_GL', 'disabled')

import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Normal
from stable_baselines3 import SAC


class GaussianPolicyNet(nn.Module):
    def __init__(self, state_dim=11, action_dim=3, hidden=64, action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x):
        mu = self.action_scale * torch.tanh(self.mu_net(x))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mu)
        return mu, std

    def log_prob(self, states, actions):
        mu, std = self.forward(states)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        return Normal(mu, std).log_prob(actions).sum(dim=-1)

    def sample_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mu, std = self.forward(state_t)
            a = Normal(mu, std).sample()
        return np.clip(a.numpy(), -self.action_scale, self.action_scale)


class RewardNet(nn.Module):
    def __init__(self, state_dim=11, gamma=0.99):
        super().__init__()
        self.net = nn.Linear(state_dim, 1, bias=True)
        self.gamma = gamma

    def base_reward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)

    def discounted_rewards(self, states: torch.Tensor) -> torch.Tensor:
        ts = torch.arange(states.size(0), dtype=torch.float32)
        return self.base_reward(states) * (self.gamma ** ts)

    def discounted_return(self, states: torch.Tensor) -> torch.Tensor:
        return self.discounted_rewards(states).sum()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.base_reward(states)


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
    def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x.float()
        y = y.float()
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt(torch.sum(x**2) * torch.sum(y**2))
        if denom < eps:
            return torch.tensor(torch.nan)
        return torch.sum(x * y) / denom

    @torch.no_grad()
    def forward(self, reward: RewardNet, trajs) -> float:
        reward.eval()
        env_returns, learned_returns = [], []
        for traj in trajs:
            env_return = torch.tensor(traj["env_rewards"], dtype=torch.float32).sum()
            env_returns.append(env_return)
            learned_return = reward.discounted_return(traj["states"])
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
    def forward(self, policy: GaussianPolicyNet, expert_trajs) -> float:
        policy.eval()
        nll = 0.0
        for traj in expert_trajs:
            log_probs = policy.log_prob(traj["states"], traj["actions"])
            nll += -log_probs.sum().item()
        return nll / len(expert_trajs)


class OuterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, policy: GaussianPolicyNet, expert_trajs) -> torch.Tensor:
        losses = []
        for traj in expert_trajs:
            log_probs = policy.log_prob(traj["states"], traj["actions"])
            losses.append(-log_probs.sum())
        return torch.stack(losses).mean()


class InnerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, policy: GaussianPolicyNet, reward: RewardNet, trajs) -> torch.Tensor:
        losses = []
        for traj in trajs:
            states = traj["states"]
            actions = traj["actions"]
            log_probs = policy.log_prob(states, actions)
            discounted_rewards = reward.discounted_rewards(states)
            losses.append((log_probs - discounted_rewards).sum())
        return torch.stack(losses).mean()


class SB3PolicyWrapper:
    def __init__(self, model, action_scale=1.0):
        self.model = model
        self.action_scale = action_scale

    def sample_action(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def eval(self):
        pass

    def train(self):
        pass


class TTSANHD(nn.Module):
    def __init__(self, policy: GaussianPolicyNet, reward: RewardNet,
                 n_cg_steps: int = 10, reg: float = 1e-3):
        super().__init__()
        self.policy = policy
        self.reward = reward
        self.n_cg_steps = n_cg_steps
        self.reg = reg

    def score(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """grad_teta log pi_teta(t)"""
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def reinforce_grad(self, trajs) -> torch.Tensor:
        """grad_teta L_inner via REINFORCE with baseline"""
        d = sum(p.numel() for p in self.policy.parameters())
        ells = []
        for traj in trajs:
            s, a = traj["states"], traj["actions"]
            T = len(s)
            gammas = torch.tensor([self.reward.gamma ** t for t in range(T)])
            with torch.no_grad():
                ell = (gammas * (self.policy.log_prob(s, a) - self.reward(s))).sum().item()
            ells.append(ell)

        baseline = float(np.mean(ells))
        grad = torch.zeros(d)
        for traj, ell in zip(trajs, ells):
            s, a = traj["states"], traj["actions"]
            s_theta = self.score(s, a)
            grad += (ell - baseline) * s_theta
        return grad / len(trajs)

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        """d L_outer/d_teta = -E[S_teta(t)] over expert trajectories"""
        d = sum(p.numel() for p in self.policy.parameters())
        grad = torch.zeros(d)
        for traj in expert_trajs:
            s, a = traj["states"], traj["actions"]
            grad += self.score(s, a)
        return -grad / len(expert_trajs)

    def cross_derivative(self, trajs) -> torch.Tensor:
        """d^2 L_inner/ (d_phi * d_teta) = -E[S_teta * delta_phi R_phi^(T)]"""
        d_theta = sum(p.numel() for p in self.policy.parameters())
        d_phi = sum(p.numel() for p in self.reward.parameters())
        cross = torch.zeros(d_theta, d_phi)
        for traj in trajs:
            s, a = traj["states"], traj["actions"]
            s_theta = self.score(s, a)
            reward_sum = self.reward.discounted_return(s)
            grads_phi = torch.autograd.grad(reward_sum, self.reward.parameters())
            g_phi = flat_grad(grads_phi)
            cross += torch.outer(s_theta, g_phi)
        return -cross / len(trajs)

    def _fisher_vector_product(self, trajs, u: torch.Tensor) -> torch.Tensor:
        """F*u where F = E[S_teta S_teta^(T)] + reg·I"""
        result = torch.zeros_like(u)
        for traj in trajs:
            s, a = traj["states"], traj["actions"]
            s_theta = self.score(s, a)
            result += s_theta * (s_theta @ u)
        result /= len(trajs)
        return result + self.reg * u

    def _conjugate_gradient_solve(self, trajs, g: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
        """Solve F*v = g via conjugate gradient"""
        v = torch.zeros_like(g)
        r = g.clone()
        p = r.clone()
        r_dot_r = (r * r).sum()
        if r_dot_r < tol:
            return v
        for _ in range(self.n_cg_steps):
            Fp = self._fisher_vector_product(trajs, p)
            pFp = (p * Fp).sum()
            if pFp <= 0:
                break
            alpha = r_dot_r / pFp
            v = v + alpha * p
            r = r - alpha * Fp
            new_r_dot_r = (r * r).sum()
            if new_r_dot_r < tol:
                break
            beta = new_r_dot_r / r_dot_r
            p = r + beta * p
            r_dot_r = new_r_dot_r
        return v

    def forward(self, expert_trajs, agent_trajs) -> torch.Tensor:
        """dL_outer/d_phi = -H^T F^{-1} g via CG"""
        g = self.outer_grad(expert_trajs)
        v = self._conjugate_gradient_solve(agent_trajs, g)
        H = self.cross_derivative(agent_trajs)
        return -H.T @ v


def flat_grad(grads):
    return torch.cat([g.flatten() for g in grads])


def assign_flat_gradients(module: nn.Module, flat_grad_vec: torch.Tensor):
    i = 0
    for p in module.parameters():
        n = p.numel()
        grad_chunk = flat_grad_vec[i : i + n]
        if grad_chunk.numel() != n:
            raise ValueError("Flat gradient has incorrect size: not enough elements for model parameters.")
        p.grad = grad_chunk.reshape(p.shape).clone()
        i += n
    if i != flat_grad_vec.numel():
        raise ValueError("Flat gradient has incorrect size: too many elements for model parameters.")


def _safe_clip_grad(grad: torch.Tensor, max_norm: float):
    if not torch.isfinite(grad).all():
        return torch.zeros_like(grad), False
    norm = grad.norm()
    if norm.item() == 0.0:
        return grad, True
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad, True


def collect_trajectories(env, policy: GaussianPolicyNet, n: int, max_steps: int = 1000):
    trajs = []
    for _ in range(n):
        states, actions, env_rewards = [], [], []
        s, _ = env.reset()
        for _ in range(max_steps):
            a = policy.sample_action(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            states.append(torch.tensor(s, dtype=torch.float32))
            actions.append(torch.tensor(a, dtype=torch.float32))
            env_rewards.append(float(r))
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


def collect_random_trajectories(env, n: int = 50, max_steps: int = 1000):
    random_policy = GaussianPolicyNet()
    return collect_trajectories(env, random_policy, n=n, max_steps=max_steps)


def load_expert_sac(path: str) -> SB3PolicyWrapper:
    model = SAC.load(path)
    return SB3PolicyWrapper(model, action_scale=1.0)


def train_expert_sac(total_timesteps=100_000, seed=42, verbose=0):
    sb3_env = gym.make("Hopper-v5")
    model = SAC("MlpPolicy", sb3_env, verbose=verbose, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    sb3_env.close()
    return SB3PolicyWrapper(model, action_scale=1.0)


def evaluation(
    env,
    policy: GaussianPolicyNet,
    reward: RewardNet,
    expert_policy: SB3PolicyWrapper,
    n_traj: int,
    max_steps: int = 1000,
):
    print("\nStep 4: final evaluation")
    print("=" * 60)

    policy.eval()
    reward.eval()
    expert_policy.eval()

    expert_test_trajs = collect_trajectories(env, expert_policy, n=n_traj, max_steps=max_steps)
    agent_test_trajs  = collect_trajectories(env, policy,        n=n_traj, max_steps=max_steps)
    random_test_trajs = collect_trajectories(env, GaussianPolicyNet(), n=n_traj, max_steps=max_steps)

    rank_pool = expert_test_trajs + agent_test_trajs + random_test_trajs

    policy_nll        = PolicyNLL()(policy, expert_test_trajs)
    rank_corr         = RankCorr()(reward, rank_pool)
    env_reward        = float(np.mean([sum(t["env_rewards"]) for t in agent_test_trajs]))
    expert_env_reward = float(np.mean([sum(t["env_rewards"]) for t in expert_test_trajs]))

    metrics = {
        "policy_nll":        policy_nll,
        "env_reward":        env_reward,
        "expert_env_reward": expert_env_reward,
        "rank_corr":         rank_corr,
    }

    print(f"PolicyNLL  = {policy_nll:.4f}")
    print(f"EnvReward  = {env_reward:.1f} (expert = {expert_env_reward:.1f})")
    print(f"RankCorr   = {rank_corr:.4f}")

    return metrics


def ttsa_irl_hopper(
    n_iterations       = 2000,
    n_traj_per_step    = 70,
    alpha_inner        = 3e-3,
    beta_outer         = 3e-4,
    n_cg_steps         = 20,
    reg_fisher         = 0.1,
    n_expert_traj      = 500,
    n_eval_traj        = 50,
    gamma              = 0.99,
    metrics_every      = 50,
    expert_path    = "sac_hopper_expert.zip",
    seed           = 42,
):
    t_start = time.time()

    env = gym.make("Hopper-v5")
    env.reset(seed=seed)

    print("=" * 60)
    print("Step 1: loading expert")
    print(f"          path = {expert_path}")
    print("=" * 60)
    t_expert_start = time.time()
    expert_policy = load_expert_sac(expert_path)
    t_expert = time.time() - t_expert_start
    print(f"Expert loaded in {t_expert:.2f}s")

    print("\nStep 2: expert's trajectories")
    expert_trajs = collect_trajectories(env, expert_policy, n=n_expert_traj)
    avg_r   = float(np.mean([sum(t["env_rewards"]) for t in expert_trajs]))
    avg_len = float(np.mean([len(t["states"])     for t in expert_trajs]))
    print(f"Expert: {n_expert_traj} trajectories, avg reward = {avg_r:.2f}, "
          f"avg len = {avg_len:.1f}")

    policy     = GaussianPolicyNet()
    reward_net = RewardNet(gamma=gamma)
    ttsa       = TTSANHD(policy, reward_net, n_cg_steps=n_cg_steps, reg=reg_fisher)

    inner_optimizer = torch.optim.SGD(policy.parameters(),     lr=alpha_inner)
    outer_optimizer = torch.optim.SGD(reward_net.parameters(), lr=beta_outer)

    outer_loss_fn = OuterLoss()
    inner_loss_fn = InnerLoss()

    history = {
        "l_outer":      [],
        "l_inner":      [],
        "env_reward":   [],
        "hypgrad_norm": [],
        "agent_len":    [],
        "iter_time":    [],
    }

    print("\nStep 3: TTSA bilevel optimization")
    print("=" * 78)
    print(f"{'iter':>5} | {'L_outer':>10} | {'L_inner':>10} | "
          f"{'EnvR':>8} | {'len':>5} | {'|hyp|':>8} | {'t/iter':>7}")
    print("-" * 78)

    for k in range(1, n_iterations + 1):
        t_iter = time.time()
        agent_trajs = collect_trajectories(env, policy, n=n_traj_per_step)

        inner_grad = ttsa.reinforce_grad(agent_trajs)
        inner_grad, ok_in = _safe_clip_grad(inner_grad, max_norm=5.0)
        if not ok_in:
            print(f"[iter {k}] WARNING: NaN/Inf in inner_grad, skipping inner step")
        else:
            inner_optimizer.zero_grad()
            assign_flat_gradients(policy, inner_grad)
            inner_optimizer.step()

        hypgrad = ttsa(expert_trajs, agent_trajs)

        hg_norm_before_clip = (
            hypgrad.norm().item() if torch.isfinite(hypgrad).all() else float("inf")
        )
        hypgrad, ok_out = _safe_clip_grad(hypgrad, max_norm=1.0)
        if not ok_out:
            print(f"[iter {k}] WARNING: NaN/Inf in hypgrad, skipping outer step")
        else:
            outer_optimizer.zero_grad()
            assign_flat_gradients(reward_net, hypgrad)
            outer_optimizer.step()

        if k % metrics_every == 0 or k == 1:
            l_out     = outer_loss_fn(policy, expert_trajs).item()
            l_in      = inner_loss_fn(policy, reward_net, agent_trajs).item()
            agent_len = float(np.mean([len(t["states"])     for t in agent_trajs]))
            env_r     = float(np.mean([sum(t["env_rewards"]) for t in agent_trajs]))

            iter_elapsed = time.time() - t_iter
            history["l_outer"].append(l_out)
            history["l_inner"].append(l_in)
            history["hypgrad_norm"].append(hg_norm_before_clip)
            history["agent_len"].append(agent_len)
            history["env_reward"].append((k, env_r))
            history["iter_time"].append(iter_elapsed)

            print(f"{k:>5} | {l_out:>10.3f} | {l_in:>10.3f} | "
                  f"{env_r:>8.1f} | {agent_len:>5.1f} | "
                  f"{hg_norm_before_clip:>8.4f} | {iter_elapsed:>6.1f}s")

    metrics = evaluation(
        env=env,
        policy=policy,
        reward=reward_net,
        expert_policy=expert_policy,
        n_traj=n_eval_traj,
    )

    env.close()
    t_total = time.time() - t_start
    avg_iter_time = float(np.mean(history["iter_time"])) if history["iter_time"] else 0.0

    print("\n" + "=" * 60)
    print("Timing:")
    # print(f"  Expert training : {t_expert:.1f}s")
    print(f"  Expert loading  : {t_expert:.2f}s")
    print(f"  TTSA total      : {t_total - t_expert:.1f}s")
    print(f"  Avg per iter    : {avg_iter_time:.1f}s")
    print(f"  Total           : {t_total:.1f}s  ({t_total/60:.1f} min)")
    print("\nLearned reward parameters:")
    w = reward_net.net.weight.data.squeeze().tolist()
    b = reward_net.net.bias.data.item()
    print(f"  weights: {[f'{x:.3f}' for x in w]}")
    print(f"  bias:    {b:.4f}")
    print(f"  gamma:   {reward_net.gamma:.4f}")

    return policy, reward_net, history, metrics


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    policy, reward_net, history, metrics = ttsa_irl_hopper()