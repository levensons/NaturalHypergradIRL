import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import Env
from torch.distributions import Normal
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


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def distribution(self, states):
        mu, std = self.forward(states)
        return Normal(mu, std)

    def log_prob(self, states, actions):
        return self.distribution(states).log_prob(actions).sum(-1)

    def sample_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            dist = self.distribution(state_tensor)
            action = dist.sample()

        return action.numpy()


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
        ts = torch.arange(states.size(0), dtype=torch.float32)
        discounts = torch.pow(self.gamma, ts)
        return discounts * rewards

    def trajectory_return(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()


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


class InnerOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        max_grad_norm: float,
        use_baseline: bool = False,
        normalize_coef: bool = False,
        eta_min: float = 0.0,
    ):
        self.policy = policy
        self.reward = reward
        self.use_baseline = use_baseline
        self.normalize_coef = normalize_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.eta_min = eta_min

        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr)
        self.scheduler = None

    def grad(self, trajs) -> torch.Tensor:
        params = list(self.policy.parameters())

        scores = []
        coefs = []

        for traj in trajs:
            states = traj["states"]
            actions = traj["actions"]

            log_prob_sum = self.policy.log_prob(states, actions).sum()

            with torch.no_grad():
                reward_return = self.reward.trajectory_return(states, actions)
                ell = log_prob_sum.detach() - reward_return
                coef = ell + 1.0

            score = flat_grad(torch.autograd.grad(log_prob_sum, params))

            scores.append(score)
            coefs.append(coef)

        scores = torch.stack(scores)
        coefs = torch.stack(coefs).float()

        if self.use_baseline:
            coefs = coefs - coefs.mean()

        if self.normalize_coef:
            coefs = coefs / (coefs.std() + 1e-8)

        grad = (coefs.unsqueeze(1) * scores).mean(dim=0)

        return grad

    def step(self, trajs):
        grad = self.grad(trajs)

        raw_grad_norm = grad.norm().item()

        if raw_grad_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / raw_grad_norm)

        clipped_grad_norm = grad.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.policy, grad)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return {
            "inner_grad_raw": raw_grad_norm,
            "inner_grad_clip": clipped_grad_norm,
        }

    def optimize(self, env, n_steps: int, n_traj: int):
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, n_steps, eta_min=self.eta_min
        )

        pbar = tqdm(range(n_steps), desc="REINFORCE inner", leave=False)

        for inner_step in pbar:
            trajs = collect_trajectories(env, self.policy, n=n_traj)

            stats = self.step(trajs)

            with torch.no_grad():
                avg_len = np.mean([len(t["states"]) for t in trajs])
                avg_env_ret = np.mean([sum(t["env_rewards"]) for t in trajs])

                inner_loss = InnerLoss()(self.policy, self.reward, trajs).item()

            lr_current = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                "g_raw": f"{stats['inner_grad_raw']:.2e}",
                "g_clip": f"{stats['inner_grad_clip']:.2e}",
                "loss": f"{inner_loss:.2f}",
                "len": f"{avg_len:.1f}",
                "ret": f"{avg_env_ret:.1f}",
                "lr": f"{lr_current:.1e}",
            })


class OuterOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        max_grad_norm: float,
        fisher_reg: float = 1e-2,
        gamma: float = 0.99,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg

        self.max_grad_norm = max_grad_norm
        self.raw_grad_norm = None
        self.clipped_grad_norm = None

        self.optimizer = torch.optim.SGD(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)

    def update_policy(self, policy: Policy):
        self.policy = policy

    def score(self, states, actions) -> torch.Tensor:
        log_prob_sum = self.policy.log_prob(states, actions).sum()
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
            grad += self.score(states, actions)

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

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)
        outer_grad = self.outer_grad(expert_trajs)
        cross = self.cross_derivative(agent_trajs)

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
                f"cross_norm={cross.norm().item():.3e}"
            )

        fisher_inv_outer_grad = torch.linalg.solve(fisher, outer_grad)
        hypergrad = -torch.einsum("tp,t->p", cross, fisher_inv_outer_grad)
        return hypergrad

    def step(self, expert_trajs, agent_trajs) -> torch.Tensor:
        hypergrad = self.hypergradient(expert_trajs, agent_trajs)

        self.raw_grad_norm = hypergrad.norm().item()
        if self.raw_grad_norm > self.max_grad_norm:
            hypergrad = hypergrad * (self.max_grad_norm / self.raw_grad_norm)

        self.clipped_grad_norm = hypergrad.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.reward, hypergrad)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return hypergrad


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


def load_irl_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")

    policy = Policy(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
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
    n_agent_inner_traj: int,
    n_agent_outer_traj: int,
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
    policy = None

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=lr_outer,
        fisher_reg=0.01,
        max_grad_norm=1.0,
        gamma=0.99,
    )

    outer_loss_fn = OuterLoss()
    inner_loss_fn = InnerLoss()
    rank_corr_fn = RankCorr()
    policy_nll_fn = PolicyNLL()

    history = {
        "l_outer": [],
        "l_inner": [],
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
        "fisher_reinforce_hopper_checkpoint.pt",
    )

    best_l_outer = float("inf")

    print("\nStep 3: bilevel optimization")
    print("=" * 190)
    print(
        f"{'Step':>5} | "
        f"{'L_outer':>10} | "
        f"{'L_inner':>10} | "
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
        policy = Policy(state_dim, action_dim)
        inner_optimizer = InnerOptimizer(
            reward=reward,
            policy=policy,
            lr=lr_inner,
            use_baseline=True,
            normalize_coef=True,
            max_grad_norm=1.0,
            eta_min=1e-6
        )
        inner_optimizer.optimize(env, n_inner_steps, n_agent_inner_traj)

        agent_trajs = collect_trajectories(env, policy, n_agent_outer_traj)
        outer_optimizer.update_policy(policy)
        outer_optimizer.step(expert_trajs, agent_trajs)

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss_fn(policy, expert_trajs).item()
        l_inner = inner_loss_fn(policy, reward, agent_trajs).item()

        agent_len = np.mean([len(t["states"]) for t in agent_trajs])
        expert_len = np.mean([len(t["states"]) for t in expert_trajs])

        agent_return = np.mean([sum(t["env_rewards"]) for t in agent_trajs])
        expert_return = np.mean([sum(t["env_rewards"]) for t in expert_trajs])

        rank_corr = rank_corr_fn(reward, expert_trajs + agent_trajs)
        policy_nll = policy_nll_fn(policy, expert_trajs)

        history["l_outer"].append(l_outer)
        history["l_inner"].append(l_inner)
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
            f"{l_inner:>10.3f} | "
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
        policy, reward, checkpoint = load_irl_checkpoint(best_checkpoint_path)
        print(f"Loaded best checkpoint from {best_checkpoint_path}")

    policy.eval()
    reward.eval()

    expert_test_trajs = collect_trajectories(
        env, expert_policy, n=n_eval_traj, max_steps=1000
    )
    agent_test_trajs = collect_trajectories(env, policy, n=n_eval_traj, max_steps=1000)
    random_test_trajs = collect_trajectories(
        env, Policy(state_dim, action_dim), n=n_eval_traj, max_steps=1000
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

    env.close()
    return policy, reward, history, metrics


if __name__ == "__main__":
    policy, reward, history, metrics = bilevel_irl(
        n_outer_steps=100,
        n_inner_steps=100,
        n_agent_inner_traj=1000,
        n_agent_outer_traj=1000,
        n_expert_traj=1000,
        n_eval_traj=1000,
        lr_outer=1e-5,
        lr_inner=1e-3,
        seed=42,
    )
