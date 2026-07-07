"""
Fisher-NHD IRL with REINFORCE inner agent for CartPole.

Usage:
    python -m src.irl.cartpole.fisher
    python -m src.irl.cartpole.fisher --config configs/cartpole.yaml
"""

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium import Env
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from src.evaluation.metrics import (
    inner_loss,
    outer_loss,
    policy_nll,
    rank_corr,
    learned_reward_stats,
)
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed, set_env_seed
from src.utils.torch import flat_grad, num_params, assign_flat_gradients, to_device
from src.utils.trajectories import (
    collect_trajectories,
    mean_trajectory_length,
    mean_trajectory_return,
    discount_weights,
)


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 16,
        n_hidden_layers: int = 1,
    ):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]

        layers.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*layers)

        self.initial_policy_state = {
            k: v.detach().clone() for k, v in self.state_dict().items()
        }

    def reset(self) -> None:
        self.load_state_dict(self.initial_policy_state)
        self.zero_grad()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(states))

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if actions.dim() > 1:
            actions = actions.squeeze(-1)

        return self.distribution(states).log_prob(actions.long())

    def sample_action(self, state, deterministic: bool = False) -> int:
        device = next(self.parameters()).device
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            dist = self.distribution(state_tensor)
            if deterministic:
                action = torch.argmax(dist.logits, dim=-1)
            else:
                action = dist.sample()

        return int(action.item())


class Reward(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 64,
        n_hidden_layers: int = 1,
        clamp_magnitude: float = 10.0,
        scale: float = 1.0,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.clamp_magnitude = clamp_magnitude

        layers = []
        in_dim = state_dim + action_dim

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def _action_features(self, actions: torch.Tensor, device) -> torch.Tensor:
        if actions.dim() > 1:
            actions = actions.squeeze(-1)

        return F.one_hot(actions.long(), num_classes=self.action_dim).to(
            device=device,
            dtype=torch.float32,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)

        action_features = self._action_features(actions, states.device)
        x = torch.cat([states, action_features], dim=-1)

        out = self.net(x).squeeze(-1)
        out = torch.clamp(out, -self.clamp_magnitude, self.clamp_magnitude)

        return self.scale * out

    def rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions)

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions).sum()


class InnerOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        discount: float,
        max_grad_norm: float,
        use_baseline: bool = False,
        normalize_coef: bool = False,
    ):
        self.reward = reward
        self.policy = policy
        self.discount = discount
        self.max_grad_norm = max_grad_norm
        self.use_baseline = use_baseline
        self.normalize_coef = normalize_coef

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

    def grad(self, trajs) -> torch.Tensor:
        params = list(self.policy.parameters())
        device = next(self.policy.parameters()).device

        scores = []
        coefs = []

        for traj in trajs:
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            weights = discount_weights(T, self.discount, device)

            log_probs = self.policy.log_prob(states, actions)
            weighted_log_prob = (weights * log_probs).sum()

            with torch.no_grad():
                reward_return = (weights * self.reward(states, actions)).sum()
                ell = weighted_log_prob.detach() - reward_return

            score = flat_grad(
                torch.autograd.grad(
                    weighted_log_prob,
                    params,
                    retain_graph=False,
                    create_graph=False,
                )
            ).detach()

            scores.append(score)
            coefs.append(ell)

        scores = torch.stack(scores)
        coefs = torch.stack(coefs).float()

        if self.use_baseline:
            coefs = coefs - coefs.mean()

        if self.normalize_coef:
            coefs = coefs / (coefs.std() + 1e-8)

        return (coefs.unsqueeze(1) * scores).mean(dim=0)

    def step(self, trajs) -> None:
        grad = self.grad(trajs)

        self.raw_grad_norm = grad.norm().item()

        if self.raw_grad_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / self.raw_grad_norm)

        self.clipped_grad_norm = grad.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.policy, grad)
        self.optimizer.step()

    def optimize(
        self,
        env: Env,
        n_steps: int,
        n_traj: int,
        max_steps: int,
        outer_step: int,
    ) -> None:
        for inner_step in tqdm(range(n_steps), desc="inner", leave=False):
            trajs = collect_trajectories(
                env=env,
                policy=self.policy,
                n=n_traj,
                max_steps=max_steps,
                desc="cartpole inner trajs",
                verbose=False,
            )

            self.step(trajs)

            mlflow.log_metrics(
                {
                    "inner/raw_grad_norm": self.raw_grad_norm,
                    "inner/clipped_grad_norm": self.clipped_grad_norm,
                },
                step=outer_step * n_steps + inner_step,
            )


class OuterOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        fisher_reg: float,
        discount: float,
        max_grad_norm: float,
        scheduler_gamma: float = 1.0,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.discount = discount
        self.max_grad_norm = max_grad_norm

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

        self.optimizer = torch.optim.Adam(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=scheduler_gamma,
        )

    def _grad_R_tail_with_discount(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.reward.parameters()).device
        T = states.size(0)

        rewards = self.reward(states, actions)  # (T,)
        weights = discount_weights(T, self.discount, device)  # (T,)
        weighted_rewards = weights * rewards

        grad_outputs = torch.eye(T, dtype=weighted_rewards.dtype, device=device)

        grads = torch.autograd.grad(
            weighted_rewards,
            self.reward.parameters(),
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )

        grads = flat_grad(grads, flat_dim=1).detach()  # (T, reward_dim)

        suffix_sums = torch.flip(
            torch.cumsum(torch.flip(grads, dims=[0]), dim=0),
            dims=[0],
        )

        return suffix_sums  # (T, reward_dim)

    def _grad_log_pi_a_s(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        T = states.size(0)

        log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
        grad_outputs = torch.eye(T, dtype=log_pi_a_s.dtype, device=device)

        grads = torch.autograd.grad(
            log_pi_a_s,
            self.policy.parameters(),
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )

        return flat_grad(grads, flat_dim=1).detach()  # (T, policy_dim)

    def fisher(self, trajs) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        F_mat = torch.zeros(
            policy_dim,
            policy_dim,
            dtype=torch.float64,
            device=device,
        )

        for traj in tqdm(trajs, desc="Fisher", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi = self._grad_log_pi_a_s(states, actions).to(torch.float64)
            weights = discount_weights(T, self.discount, device, dtype=torch.float64)

            F_mat += torch.einsum(
                "t,ti,tj->ij",
                weights,
                grad_log_pi,
                grad_log_pi,
            )

        F_mat /= len(trajs)
        F_mat = 0.5 * (F_mat + F_mat.T)
        F_mat += self.fisher_reg * torch.eye(
            policy_dim,
            dtype=torch.float64,
            device=device,
        )
        F_mat = 0.5 * (F_mat + F_mat.T)

        return F_mat

    def d_outer_d_policy(self, expert_trajs) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        out = torch.zeros(policy_dim, dtype=torch.float32, device=device)

        for traj in tqdm(expert_trajs, desc="Outer grad", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            log_pi = self.policy.log_prob(states, actions)
            weights = discount_weights(T, self.discount, device)

            weighted_log_pi = (weights * log_pi).sum()

            grad = torch.autograd.grad(
                weighted_log_pi,
                self.policy.parameters(),
                retain_graph=False,
                create_graph=False,
            )

            out += flat_grad(grad).detach()

        return -(out / len(expert_trajs))

    def d_cross_vec_product(self, trajs, v: torch.Tensor) -> torch.Tensor:
        reward_dim = num_params(self.reward)
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        v = v.to(dtype=torch.float32, device=device)

        if v.numel() != policy_dim:
            raise ValueError(f"`v` must have shape ({policy_dim},), got {tuple(v.shape)}.")

        out = torch.zeros(reward_dim, dtype=torch.float32, device=device)

        for traj in tqdm(trajs, desc="Cross vec product", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)

            grad_R_tail = self._grad_R_tail_with_discount(states, actions)
            grad_log_pi = self._grad_log_pi_a_s(states, actions)

            out += torch.einsum(
                "tr,tp,p->r",
                grad_R_tail,
                grad_log_pi,
                v,
            )

        return -(out / len(trajs))

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)
        d_outer = self.d_outer_d_policy(expert_trajs).to(dtype=torch.float64)

        fisher_inv_d_outer = torch.linalg.solve(fisher, d_outer)
        fisher_inv_d_outer = fisher_inv_d_outer.to(dtype=torch.float32)

        hypergrad = -self.d_cross_vec_product(agent_trajs, fisher_inv_d_outer)

        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(fisher)
            min_eig = eigvals.min()
            max_eig = eigvals.max()
            cond_number = max_eig / min_eig

            tqdm.write(
                f"Fisher stats | "
                f"min_eig={min_eig.item():.3e} | "
                f"max_eig={max_eig.item():.3e} | "
                f"cond={cond_number.item():.3e} | "
                f"outer_grad_norm={d_outer.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e}"
            )

            mlflow.log_metrics(
                {
                    "fisher_min_eig": min_eig.item(),
                    "fisher_max_eig": max_eig.item(),
                    "fisher_cond": cond_number.item(),
                    "outer_grad_norm": d_outer.norm().item(),
                    "hypergrad_norm_before_clip": hypergrad.norm().item(),
                }
            )

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


def train_bilevel(env: Env, config: dict, logger) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    ckpt_cfg = config["checkpoint"]

    if inner_cfg["type"] != "reinforce":
        raise ValueError(f"Expected fisher.inner.type = reinforce, got {inner_cfg['type']}")

    discount = float(reward_cfg.get("gamma", fisher_cfg.get("discount", fisher_cfg.get("gamma", 0.99))))
    scheduler_gamma = float(fisher_cfg.get("scheduler_gamma", fisher_cfg.get("gamma", 1.0)))

    mlflow.log_params(
        {
            "discount": discount,
            "lr_reward": fisher_cfg["lr_reward"],
            "fisher_reg": fisher_cfg["fisher_reg"],
            "n_outer_steps": fisher_cfg["n_outer_steps"],
            "n_inner_steps": fisher_cfg["n_inner_steps"],
            "n_agent_traj": fisher_cfg["n_agent_traj"],
            "reward_hidden": reward_cfg["hidden"],
            "policy_hidden": policy_cfg["hidden"],
            "lr_policy": inner_cfg["lr_policy"],
            "inner_type": inner_cfg["type"],
            "use_baseline": inner_cfg["use_baseline"],
            "normalize_coef": inner_cfg["normalize_coef"],
        }
    )

    data_cfg = config["data"]

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    logger.info(f"Loaded {len(expert_train_trajs)} expert train trajectories from {expert_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} expert valid trajectories from {expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} random valid trajectories from {random_valid_path}")

    n_outer_steps = int(fisher_cfg["n_outer_steps"])
    n_inner_steps = int(fisher_cfg["n_inner_steps"])
    n_agent_traj = int(fisher_cfg["n_agent_traj"])
    max_steps = int(config["env"]["max_steps"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    hidden = int(policy_cfg["hidden"])
    n_hidden_layers = int(policy_cfg["n_hidden_layers"])

    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=hidden,
        n_hidden_layers=n_hidden_layers,
    ).to(device)

    reward = Reward(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=int(reward_cfg["hidden"]),
        n_hidden_layers=int(reward_cfg.get("n_hidden_layers", 1)),
        clamp_magnitude=float(reward_cfg.get("clamp_magnitude", 10.0)),
        scale=float(reward_cfg.get("scale", 1.0)),
    ).to(device)

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        discount=discount,
        max_grad_norm=float(fisher_cfg["max_grad_norm"]),
        scheduler_gamma=scheduler_gamma,
    )

    inner_optimizer = InnerOptimizer(
        reward=reward,
        policy=policy,
        lr=float(inner_cfg["lr_policy"]),
        discount=discount,
        max_grad_norm=float(inner_cfg["max_grad_norm"]),
        use_baseline=bool(inner_cfg["use_baseline"]),
        normalize_coef=bool(inner_cfg["normalize_coef"]),
    )

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

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "fisher.pt")
    best_env_reward = float("-inf")

    arch = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "policy_hidden": hidden,
        "policy_n_hidden_layers": n_hidden_layers,
        "reward_hidden": int(reward_cfg["hidden"]),
        "reward_n_hidden_layers": int(reward_cfg.get("n_hidden_layers", 1)),
        "reward_clamp_magnitude": float(reward_cfg.get("clamp_magnitude", 10.0)),
        "reward_scale": float(reward_cfg.get("scale", 1.0)),
        "discount": discount,
        "method": "fisher",
        "agent": "reinforce",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    def eval_reward_stats() -> None:
        expert_reward_stats = learned_reward_stats(
            reward,
            expert_valid_trajs,
            outer_optimizer.discount,
        )
        random_reward_stats = learned_reward_stats(
            reward,
            random_valid_trajs,
            outer_optimizer.discount,
        )

        expert_learned_return = expert_reward_stats["return_mean"]
        random_learned_return = random_reward_stats["return_mean"]
        expert_random_learned_diff = expert_learned_return - random_learned_return

        expert_learned_step_mean = expert_reward_stats["step_mean"]
        random_learned_step_mean = random_reward_stats["step_mean"]

        logger.info(
            f"   [outer] "
            f"expert_ret={expert_learned_return:.3f} "
            f"random_ret={random_learned_return:.3f} "
            f"diff={expert_random_learned_diff:.3f} "
            f"expert_step={expert_learned_step_mean:.4f} "
            f"random_step={random_learned_step_mean:.4f}"
        )

        mlflow.log_metrics(
            {
                "reward/expert_learned_return": expert_learned_return,
                "reward/random_learned_return": random_learned_return,
                "reward/expert_random_learned_diff": expert_random_learned_diff,
                "reward/expert_step_mean": expert_learned_step_mean,
                "reward/random_step_mean": random_learned_step_mean,
            }
        )

    def log_and_checkpoint(outer_step: int, agent_trajs) -> None:
        nonlocal best_env_reward

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss(policy, expert_train_trajs, outer_optimizer.discount)
        if hasattr(l_outer, "item"):
            l_outer = l_outer.item()

        l_inner = inner_loss(policy, reward, agent_trajs, discount=1.0)
        if hasattr(l_inner, "item"):
            l_inner = l_inner.item()

        agent_len = mean_trajectory_length(agent_trajs)
        expert_len = mean_trajectory_length(expert_train_trajs)

        agent_ret = mean_trajectory_return(agent_trajs)
        expert_ret = mean_trajectory_return(expert_train_trajs)

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
                path=best_checkpoint_path,
                policy=policy,
                reward=reward,
                arch=arch,
                outer_step=outer_step,
                best_env_reward=best_env_reward,
            )

        row = (
            f"{outer_step:>5} | {l_outer:>10.3f} | {l_inner:>10.3f} | "
            f"{agent_len:>10.1f} | {expert_len:>10.1f} | "
            f"{agent_ret:>10.1f} | {expert_ret:>10.1f} | "
            f"{rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )

        logger.info(row)

        mlflow.log_metrics(
            {
                "outer_loss": l_outer,
                "inner_loss": l_inner,
                "agent_return": agent_ret,
                "expert_return": expert_ret,
                "agent_length": agent_len,
                "expert_length": expert_len,
                "rank_corr": rank_corr_val,
                "policy_nll": policy_nll_val,
                "hypergrad_norm": raw_hypgrad_norm,
                "hypergrad_norm_clipped": clipped_hypgrad_norm,
                "lr_reward": lr_outer_current,
            },
            step=outer_step,
        )

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'L_inner':>10} | "
        f"{'agent_len':>10} | {'expert_len':>10} | "
        f"{'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | "
        f"{'hyp_raw':>10} | {'hyp_clip':>10} | {'lr_outer':>12}"
    )

    logger.info(header)

    eval_reward_stats()

    inner_optimizer.optimize(
        env=env,
        n_steps=n_inner_steps,
        n_traj=n_agent_traj,
        max_steps=max_steps,
        outer_step=0,
    )

    agent_trajs = collect_trajectories(
        env=env,
        policy=policy,
        n=n_agent_traj,
        max_steps=max_steps,
        desc="cartpole outer agent trajs",
        verbose=False,
    )

    log_and_checkpoint(outer_step=0, agent_trajs=agent_trajs)

    for outer_step in range(1, n_outer_steps + 1):
        outer_optimizer.step(expert_train_trajs, agent_trajs)
        eval_reward_stats()

        # policy.reset()

        inner_optimizer.optimize(
            env=env,
            n_steps=n_inner_steps,
            n_traj=n_agent_traj,
            max_steps=max_steps,
            outer_step=outer_step,
        )

        agent_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_traj,
            max_steps=max_steps,
            desc="cartpole outer agent trajs",
            verbose=False,
        )

        log_and_checkpoint(outer_step=outer_step, agent_trajs=agent_trajs)

    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD IRL REINFORCE — CartPole")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("cartpole", args.config)
    config = load_config(config_path)

    fisher_cfg = config["fisher"]
    log_cfg = config["logging"]

    log_dir = log_cfg["log_dir"]
    logger = get_logger("fisher_cartpole", log_dir=log_dir)

    set_random_seed(int(fisher_cfg["random_seed"]))

    env = gym.make(config["env"]["id"])
    set_env_seed(env, int(fisher_cfg["env_seed"]))

    try:
        logger.info("=== Fisher-NHD CartPole REINFORCE ===")

        mlflow.set_experiment("fisher")
        with mlflow.start_run(run_name="cartpole"):
            history = train_bilevel(env, config, logger)

        report_path = Path(log_cfg["report_dir"]) / "fisher_reinforce_cartpole_history.json"
        save_history(history, str(report_path))
        logger.info(f"History saved to {report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()