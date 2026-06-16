"""
Fisher-NHD IRL для CartPole (дискретная среда).

Запуск:
  python -m irl.cartpole.fisher_cartpole --env cartpole
  python -m irl.cartpole.fisher_cartpole --config config/cartpole.yaml
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Env
from torch.distributions import Categorical
from tqdm import tqdm

from evaluation.metrics import inner_loss, outer_loss, policy_nll, rank_corr
from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.config import load_config, resolve_config_path, set_seed
from src.common.logging_utils import get_logger, save_history


def flat_grad(params):
    return torch.cat([p.flatten() for p in params])


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def assign_flat_gradients(module: nn.Module, flat_grad: torch.Tensor):
    i = 0
    for p in module.parameters():
        n = p.numel()
        grad_chunk = flat_grad[i : i + n]
        if grad_chunk.numel() != n:
            raise ValueError("Flat gradient has incorrect size: not enough elements.")
        p.grad = grad_chunk.reshape(p.shape).clone()
        i += n
    if i != flat_grad.numel():
        raise ValueError("Flat gradient has incorrect size: too many elements.")


def get_env_dims(env: Env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return state_dim, action_dim


class WarmupLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, gamma: float = 0.95,
                 min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        decay_step = step - self.warmup_steps
        return [max(self.min_lr, base_lr * (self.gamma**decay_step)) for base_lr in self.base_lrs]


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, n_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return self.net(states)

    def distribution(self, states):
        return Categorical(logits=self.forward(states))

    def log_prob(self, states, actions):
        return self.distribution(states).log_prob(actions.long())

    def sample_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.distribution(state_tensor).sample()
        return int(action.item())


class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Linear(state_dim, 1)
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, states: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        return self.net(states).squeeze(-1)

    def discounted_rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        rewards = self.forward(states, actions)
        ts = torch.arange(states.size(0), dtype=torch.float32, device=states.device)
        return torch.pow(self.gamma, ts) * rewards

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()


class SB3PolicyWrapper:
    def __init__(self, model):
        self.model = model

    def sample_action(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)

    def eval(self):
        pass


def collect_trajectories(env: Env, policy, n: int, max_steps=500):
    trajs = []
    for _ in tqdm(range(n), desc="collect trajs", leave=False):
        states, actions, env_rewards = [], [], []
        state, _ = env.reset()
        for _ in range(max_steps):
            action = policy.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(torch.tensor(state, dtype=torch.float32))
            actions.append(torch.tensor(action, dtype=torch.long))
            env_rewards.append(float(reward))
            state = next_state
            if terminated or truncated:
                break
        trajs.append({"states": torch.stack(states), "actions": torch.stack(actions),
                      "env_rewards": env_rewards})
    return trajs


class InnerOptimizer:
    def __init__(self, reward: Reward, policy: Policy, lr: float, max_grad_norm: float,
                 use_baseline: bool = False, normalize_coef: bool = False):
        self.policy = policy
        self.reward = reward
        self.use_baseline = use_baseline
        self.normalize_coef = normalize_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr)
        self.scheduler = None

    def grad(self, trajs) -> torch.Tensor:
        params = list(self.policy.parameters())
        gamma_val = float(self.reward.gamma)
        scores, coefs = [], []
        for traj in trajs:
            states, actions = traj["states"], traj["actions"]
            T = states.size(0)
            gammas = torch.pow(torch.tensor(gamma_val), torch.arange(T, dtype=torch.float32))
            log_probs = self.policy.log_prob(states, actions)
            log_prob_discounted = (gammas * log_probs).sum()
            with torch.no_grad():
                reward_return = self.reward.trajectory_return(states, actions)
                ell = (gammas * log_probs.detach()).sum() - reward_return

            score = flat_grad(torch.autograd.grad(log_prob_discounted, params))
            scores.append(score)
            coefs.append(ell)
        scores = torch.stack(scores)
        coefs = torch.stack(coefs).float()
        coefs = coefs - coefs.mean()
        if self.normalize_coef:
            coefs = coefs / (coefs.std() + 1e-8)
        return (coefs.unsqueeze(1) * scores).mean(dim=0)

    def step(self, trajs):
        grad = self.grad(trajs)
        grad_norm = grad.norm().item()
        if grad_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / grad_norm)
        self.optimizer.zero_grad()
        assign_flat_gradients(self.policy, grad)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def optimize(self, env, n_steps: int, n_traj: int):
        # Константный LR на внутреннем цикле (без cosine-затухания до 1e-6)
        self.scheduler = None
        for _ in tqdm(range(n_steps), desc="inner", leave=False):
            trajs = collect_trajectories(env, self.policy, n_traj)
            self.step(trajs)


class OuterOptimizer:
    def __init__(self, reward: Reward, policy: Policy, lr: float, max_grad_norm: float,
                 warmup_steps: int = 10, min_lr: float = 0.0, fisher_reg: float = 1e-2,
                 gamma: float = 0.99):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.max_grad_norm = max_grad_norm
        self.raw_grad_norm = None
        self.clipped_grad_norm = None
        self.optimizer = torch.optim.SGD(self.reward.parameters(), lr=lr)
        self.scheduler = WarmupLR(self.optimizer, warmup_steps, gamma, min_lr=min_lr)

    def update_policy(self, policy: Policy):
        self.policy = policy

    def score(self, states, actions) -> torch.Tensor:
        log_prob_sum = self.policy.log_prob(states, actions).sum()
        grads = torch.autograd.grad(log_prob_sum, self.policy.parameters())
        return flat_grad(grads)

    def fisher(self, trajs) -> torch.Tensor:
        d = num_params(self.policy)
        F_mat = torch.zeros(d, d)
        for traj in trajs:
            s = self.score(traj["states"], traj["actions"])
            F_mat += torch.outer(s, s)
        F_mat /= len(trajs)
        F_mat += self.fisher_reg * torch.eye(d)
        return F_mat

    def outer_grad(self, expert_trajs) -> torch.Tensor:
        d = num_params(self.policy)
        grad = torch.zeros(d)
        for traj in expert_trajs:
            grad += self.score(traj["states"], traj["actions"])
        return -grad / len(expert_trajs)

    def cross_derivative(self, trajs) -> torch.Tensor:
        d_theta = num_params(self.policy)
        d_phi = num_params(self.reward)
        H = torch.zeros(d_theta, d_phi)
        for traj in trajs:
            states, actions = traj["states"], traj["actions"]
            s_theta = self.score(states, actions)
            reward_return = self.reward.trajectory_return(states, actions)
            grads_phi = torch.autograd.grad(reward_return, self.reward.parameters())
            g_phi = flat_grad(grads_phi)
            H += torch.outer(s_theta, g_phi)
        return -H / len(trajs)

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)
        outer_grad = self.outer_grad(expert_trajs)
        cross = self.cross_derivative(agent_trajs)
        fisher_inv_outer_grad = torch.linalg.solve(fisher, outer_grad)
        return -torch.einsum("tp,t->p", cross, fisher_inv_outer_grad)

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


def train_bilevel(env, expert_trajs, config: dict, logger=None) -> dict:
    """Двухуровневая оптимизация Fisher-NHD. Алгоритм не изменён."""
    train_cfg = config["training"]
    inner_cfg = config.get("inner_agent", {})
    outer_cfg = config.get("outer_optimizer", {})
    policy_cfg = config.get("policy", {})
    reward_cfg = config.get("reward", {})
    ckpt_cfg = config.get("checkpoint", {})
    log_cfg  = config.get("logging", {})
    data_cfg = config.get("data", {})

    expert_valid_trajs = torch.load(
        data_cfg.get("expert_valid_trajs", "data/cartpole/expert_valid_trajs.pt"),
        map_location="cpu", weights_only=False,
    )
    random_valid_trajs = torch.load(
        data_cfg.get("random_valid_trajs", "data/cartpole/random_valid_trajs.pt"),
        map_location="cpu", weights_only=False,
    )

    n_outer_steps = int(train_cfg["n_outer_steps"])
    n_inner_steps = int(train_cfg["n_inner_steps"])
    n_agent_traj  = int(train_cfg["n_agent_traj"])

    state_dim, action_dim = get_env_dims(env)

    policy = Policy(
        state_dim=state_dim, action_dim=action_dim,
        hidden=int(policy_cfg.get("hidden", 64)),
        n_hidden_layers=int(policy_cfg.get("n_hidden_layers", 2)),
    )
    reward = Reward(
        state_dim=state_dim, action_dim=action_dim,
        hidden=int(reward_cfg.get("hidden", 64)),
        gamma=float(reward_cfg.get("gamma", 0.99)),
    )

    outer_optimizer = OuterOptimizer(
        reward=reward, policy=policy,
        lr=float(train_cfg["lr_outer"]),
        fisher_reg=float(outer_cfg.get("fisher_reg", 1e-2)),
        max_grad_norm=float(outer_cfg.get("max_grad_norm", 100.0)),
        warmup_steps=int(outer_cfg.get("warmup_steps", 5)),
        min_lr=float(outer_cfg.get("min_lr", 1e-6)),
        gamma=float(outer_cfg.get("gamma", 0.99)),
    )
    inner_optimizer = InnerOptimizer(
        reward=reward, policy=policy,
        lr=float(train_cfg["lr_inner"]),
        max_grad_norm=float(inner_cfg.get("max_grad_norm", 10.0)),
        use_baseline=bool(inner_cfg.get("use_baseline", False)),
        normalize_coef=bool(inner_cfg.get("normalize_coef", False)),
    )

    history = {
        "l_outer": [], "l_inner": [], "agent_len": [], "expert_len": [],
        "agent_return": [], "expert_return": [], "rank_corr": [], "policy_nll": [],
        "raw_hypgrad_norm": [], "clipped_hypgrad_norm": [], "lr_outer": [],
    }

    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/cartpole"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = str(ckpt_dir / "fisher_reinforce.pt")
    best_env_reward = float("-inf")

    arch = {
        "state_dim": state_dim, "action_dim": action_dim,
        "policy_hidden": int(policy_cfg.get("hidden", 64)),
        "policy_n_hidden_layers": int(policy_cfg.get("n_hidden_layers", 2)),
        "reward_hidden": int(reward_cfg.get("hidden", 64)),
        "reward_gamma": float(reward_cfg.get("gamma", 0.99)),
        "method": "fisher", "agent": "reinforce",
        "env_name": config["env"]["name"], "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'L_inner':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )
    if logger:
        logger.info(header)
    else:
        print(header)

    for outer_step in range(1, n_outer_steps + 1):
        inner_optimizer.optimize(env, n_inner_steps, n_agent_traj)
        agent_trajs = collect_trajectories(env, policy, n_agent_traj)
        outer_optimizer.step(expert_trajs, agent_trajs)

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss(policy, expert_trajs).item()
        l_inner = inner_loss(policy, reward, agent_trajs).item()
        agent_len = np.mean([len(t["states"]) for t in agent_trajs])
        expert_len = np.mean([len(t["states"]) for t in expert_trajs])
        agent_ret = np.mean([sum(t["env_rewards"]) for t in agent_trajs])
        expert_ret = np.mean([sum(t["env_rewards"]) for t in expert_trajs])
        rank_corr_val = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
        policy_nll_val = policy_nll(policy, expert_valid_trajs)

        history["l_outer"].append(l_outer)
        history["l_inner"].append(l_inner)
        history["agent_len"].append(agent_len)
        history["expert_len"].append(expert_len)
        history["agent_return"].append(agent_ret)
        history["expert_return"].append(expert_ret)
        history["rank_corr"].append(rank_corr_val)
        history["policy_nll"].append(policy_nll_val)
        history["raw_hypgrad_norm"].append(raw_hypgrad_norm)
        history["clipped_hypgrad_norm"].append(clipped_hypgrad_norm)
        history["lr_outer"].append(lr_outer_current)

        if agent_ret > best_env_reward:
            best_env_reward = agent_ret
            save_checkpoint(
                path=best_checkpoint_path, policy=policy, reward=reward, arch=arch,
                outer_step=outer_step, best_env_reward=best_env_reward,
            )

        row = (
            f"{outer_step:>5} | {l_outer:>10.3f} | {l_inner:>10.3f} | "
            f"{agent_len:>10.1f} | {expert_len:>10.1f} | {agent_ret:>10.1f} | "
            f"{expert_ret:>10.1f} | {rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )
        if logger:
            logger.info(row)
        else:
            print(row)

    return history


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fisher-NHD IRL — CartPole")
    p.add_argument("--env", choices=["cartpole"], default="cartpole")
    p.add_argument("--config", default=None, help="Путь к config YAML. Перекрывает --env.")
    return p.parse_args()


def main():
    args = parse()
    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)

    train_cfg  = config["training"]
    log_cfg = config.get("logging", {})
    data_cfg = config.get("data", {})

    set_seed(int(train_cfg.get("seed", 42)))

    log_dir = log_cfg.get("log_dir", "logs")
    logger  = get_logger("fisher_cartpole", log_dir=log_dir)

    env = gym.make(config["env"]["id"])
    env.reset(seed=int(train_cfg.get("seed", 42)))

    # Загружаем экспертные траектории из data/ (не собираем заново)
    expert_train_path = data_cfg.get("expert_train_trajs", "data/cartpole/expert_train_trajs.pt")
    all_expert_trajs = torch.load(expert_train_path, map_location="cpu", weights_only=False)

    expert_trajs = all_expert_trajs
    logger.info(f"Загружено {len(expert_trajs)} экспертных траекторий из {expert_train_path}")

    logger.info("=== Fisher-NHD CartPole ===")
    history = train_bilevel(env, expert_trajs, config, logger=logger)

    report_path = (
        Path(log_cfg.get("report_dir", "reports")) / "fisher_cartpole_history.json"
    )
    save_history(history, str(report_path))
    logger.info(f"История сохранена в {report_path}")

    env.close()


if __name__ == "__main__":
    main()