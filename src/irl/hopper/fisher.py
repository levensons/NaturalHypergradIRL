"""
Fisher-NHD IRL with SAC inner agent for Hopper.

Usage:
    python -m src.irl.hopper.fisher
    python -m src.irl.hopper.fisher --config configs/hopper.yaml
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import gymnasium as gym
from gymnasium import Env
import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import inner_loss, outer_loss, policy_nll, rank_corr
from src.agents.sac import SACInnerOptimizer
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import get_env_dims
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed, set_env_seed
from src.utils.torch import flat_grad, num_params, assign_flat_gradients, to_device, suffix_sum
from src.utils.trajectories import (
    collect_trajectories,
    mean_trajectory_length,
    mean_trajectory_return,
    discount_weights,
)


class Reward(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_hidden_layers=2,
        hidden=64,
        clamp_magnitude=10.0,
    ):
        super().__init__()

        self.clamp_magnitude = clamp_magnitude

        self.first_fc = nn.Linear(state_dim + action_dim, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU()) for _ in range(n_hidden_layers - 1)]
        )
        self.last_fc = nn.Linear(hidden, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = self.first_fc(torch.cat([states, actions], dim=-1))
        for block in self.blocks:
            out = block(out)
        out = self.last_fc(out).squeeze(-1)
        out = torch.clamp(out, -self.clamp_magnitude, self.clamp_magnitude)
        return out

    def rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions)

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions).sum()


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        hidden: int = 64,
        n_hidden_layers: int = 2,
        log_std_max=2,
        log_std_min=-5,
    ):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]

        self.net = nn.Sequential(*layers)
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
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, states: torch.Tensor, eps: float = 1e-6):
        mean, log_std = self.forward(states)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1.0 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action

    @staticmethod
    def atanh(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def action_to_normalized(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_bias) / self.action_scale

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        mean, log_std = self.forward(states)
        std = log_std.exp()

        normalized_action = torch.clamp(
            self.action_to_normalized(actions),
            -1.0 + eps,
            1.0 - eps,
        )
        pre_tanh_action = self.atanh(normalized_action)

        normal = torch.distributions.Normal(mean, std)

        log_prob = normal.log_prob(pre_tanh_action)
        correction = torch.log(self.action_scale * (1.0 - normalized_action.pow(2)) + eps)

        return (log_prob - correction).sum(dim=-1)

    def sample_action(self, state, deterministic: bool = False):
        device = next(self.parameters()).device
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.get_action(state_tensor)
            else:
                action, _, _ = self.get_action(state_tensor)

        return action.squeeze(0).cpu().numpy()


class OuterOptimizer:
    def __init__(
        self,
        reward: Reward,
        policy: Policy,
        lr: float,
        fisher_reg: float,
        discount: float,
        max_grad_norm=None,
        scheduler_gamma: float = 1.0,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.discount = discount
        self.max_grad_norm = max_grad_norm
        self.scheduler_gamma = scheduler_gamma

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

        self.optimizer = torch.optim.Adam(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, scheduler_gamma)

    def _grad_R_tail_with_discount(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        device = next(self.reward.parameters()).device
        T = states.size(0)

        r_a_s_t = self.reward(states, actions)  # (T,)
        weights = discount_weights(T, self.discount, device)  # (T,)
        r_a_s_t = weights * r_a_s_t

        grad_outputs = torch.eye(T, dtype=r_a_s_t.dtype, device=device)  # (T, T)

        grads = torch.autograd.grad(
            r_a_s_t,
            self.reward.parameters(),
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )

        grads = flat_grad(grads, flat_dim=1).detach()  # (T, reward_dim)
        suffix_sums = suffix_sum(grads, dim=0)  # (T, reward_dim)
        return suffix_sums

    def _grad_log_pi_a_s(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        T = states.size(0)

        log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
        grad_outputs = torch.eye(T, dtype=log_pi_a_s.dtype, device=device)  # (T, T)

        grads = torch.autograd.grad(
            log_pi_a_s,
            self.policy.parameters(),
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )

        grads = flat_grad(grads, flat_dim=1).detach()  # (T, policy_dim)
        return grads

    def fisher(self, trajs) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        F = torch.zeros(policy_dim, policy_dim, dtype=torch.float64, device=device)

        for traj in tqdm(trajs, desc="Fisher", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions).to(torch.float64)  # (T, policy_dim)
            weights = discount_weights(T, self.discount, device, dtype=torch.float64)  # (T,)
            F += torch.einsum("t,ti,tj->ij", weights, grad_log_pi_a_s, grad_log_pi_a_s)  # (policy_dim, policy_dim)

        F /= len(trajs)
        F = 0.5 * (F + F.T)
        F += self.fisher_reg * torch.eye(policy_dim, dtype=torch.float64, device=device)
        F = 0.5 * (F + F.T)
        return F

    def d_outer_d_policy(self, expert_trajs) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        E = torch.zeros(policy_dim, dtype=torch.float32, device=device)

        for traj in tqdm(expert_trajs, desc="Outer grad", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
            weights = discount_weights(T, self.discount, device)  # (T,)
            sum_weighted_log_pi_a_s = (weights * log_pi_a_s).sum()  # (1,)

            grad = torch.autograd.grad(
                sum_weighted_log_pi_a_s, self.policy.parameters(), retain_graph=False, create_graph=False
            )

            grad = flat_grad(grad).detach()  # (policy_dim,)
            E += grad

        return -(E / len(expert_trajs))

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

            grad_R_tail_with_discount = self._grad_R_tail_with_discount(states, actions)  # (T, reward_dim)
            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions)  # (T, policy_dim)
            out += torch.einsum("tr,tp,p->r", grad_R_tail_with_discount, grad_log_pi_a_s, v)  # (reward_dim,)

        return -(out / len(trajs))

    def hypergradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        fisher = self.fisher(agent_trajs)  # (policy_dim, policy_dim)
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs).to(dtype=torch.float64)  # (policy_dim,)

        fisher_inv_d_outer_d_policy = torch.linalg.solve(fisher, d_outer_d_policy)  # (policy_dim,)
        fisher_inv_d_outer_d_policy = fisher_inv_d_outer_d_policy.to(dtype=torch.float32)

        hypergrad = -self.d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # (reward_dim,)

        # with torch.no_grad():
        #     eigvals = torch.linalg.eigvalsh(fisher)
        #     min_eig = eigvals.min()
        #     max_eig = eigvals.max()
        #     cond_number = max_eig / min_eig
        #     print(
        #         f"Fisher stats | "
        #         f"min_eig={min_eig.item():.3e} | "
        #         f"max_eig={max_eig.item():.3e} | "
        #         f"cond={cond_number.item():.3e} | "
        #         f"outer_grad_norm={grad_outer.norm().item():.3e} | "
        #         f"hypergrad_norm={hypergrad.norm().item():.3e}"
        #     )

        return hypergrad

    def step(self, expert_trajs, agent_trajs) -> torch.Tensor:
        hypergradient = self.hypergradient(expert_trajs, agent_trajs)

        self.raw_grad_norm = hypergradient.norm().item()
        if self.max_grad_norm is not None and self.raw_grad_norm > self.max_grad_norm:
            hypergradient = hypergradient * (self.max_grad_norm / self.raw_grad_norm)
        self.clipped_grad_norm = hypergradient.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.reward, hypergradient)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return hypergradient


def make_sac_inner_optimizer(
    sac_env: Env,
    env: Env,
    reward: Reward,
    policy: Policy,
    state_dim: int,
    action_dim: int,
    fisher_cfg: dict,
    device: str | torch.device = "cpu",
):
    inner_cfg = fisher_cfg["inner"]
    sac_cfg = inner_cfg["sac"]

    return SACInnerOptimizer(
        env=sac_env,
        reward=reward,
        policy=policy,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        lr=float(sac_cfg["lr"]),
        buffer_size=int(sac_cfg["buffer_size"]),
        batch_size=int(sac_cfg["batch_size"]),
        learning_starts=int(sac_cfg["learning_starts"]),
        gamma=float(sac_cfg["gamma"]),
        polyak=float(sac_cfg["polyak"]),
        alpha=float(sac_cfg["alpha"]),
        autotune=bool(sac_cfg["autotune"]),
        update_every=int(sac_cfg["update_every"]),
        update_num=int(sac_cfg["update_num"]),
        q_hidden=int(sac_cfg["q_hidden"]),
        q_n_hidden_layers=int(sac_cfg["q_n_hidden_layers"]),
        device=device,
    )


def train_bilevel(env, config: dict, logger) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    ckpt_cfg = config["checkpoint"]

    if inner_cfg["type"] != "sac":
        raise ValueError(f"Expected fisher.inner.type = sac, got {inner_cfg['type']}")

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

    state_dim, action_dim = get_env_dims(env)

    hidden = int(policy_cfg["hidden"])
    n_layers = int(policy_cfg["n_hidden_layers"])

    reward = Reward(
        state_dim=state_dim,
        action_dim=action_dim,
        n_hidden_layers=int(reward_cfg["n_hidden_layers"]),
        hidden=int(reward_cfg["hidden"]),
        clamp_magnitude=float(reward_cfg["clamp_magnitude"]),
    ).to(device)

    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden=hidden,
        n_hidden_layers=n_layers,
        log_std_min=float(policy_cfg["log_std_min"]),
        log_std_max=float(policy_cfg["log_std_max"]),
    ).to(device)

    initial_policy_state = {
        k: v.detach().clone()
        for k, v in policy.state_dict().items()
    }

    outer_optimizer = OuterOptimizer(
        reward=reward,
        policy=policy,
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        discount=float(fisher_cfg["discount"]),
        max_grad_norm=float(fisher_cfg["max_grad_norm"]),
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
    )

    sac_env = gym.make(config["env"]["id"])
    set_env_seed(sac_env, int(inner_cfg["sac_env_seed"]))

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

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "fisher.pt")
    best_env_reward = float("-inf")

    arch = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "policy_hidden": hidden,
        "policy_n_hidden_layers": n_layers,
        "reward_n_hidden_layers": int(reward_cfg["n_hidden_layers"]),
        "reward_hidden": int(reward_cfg["hidden"]),
        "reward_clamp_magnitude": float(reward_cfg["clamp_magnitude"]),
        "log_std_min": float(policy_cfg["log_std_min"]),
        "log_std_max": float(policy_cfg["log_std_max"]),
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "method": "fisher",
        "agent": "sac",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    def log_and_checkpoint(outer_step: int, agent_trajs):
        nonlocal best_env_reward

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss(policy, expert_train_trajs, outer_optimizer.discount)

        agent_len = mean_trajectory_length(agent_trajs)
        expert_len = mean_trajectory_length(expert_train_trajs)

        agent_ret = mean_trajectory_return(agent_trajs)
        expert_ret = mean_trajectory_return(expert_train_trajs)

        rank_corr_val = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
        policy_nll_val = policy_nll(policy, expert_valid_trajs)

        history["l_outer"].append(l_outer)
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
            f"{outer_step:>5} | {l_outer:>10.3f} | {agent_len:>10.1f} | "
            f"{expert_len:>10.1f} | {agent_ret:>10.1f} | {expert_ret:>10.1f} | "
            f"{rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )

        logger.info(row)

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )

    logger.info(header)

    inner_optimizer = make_sac_inner_optimizer(
        sac_env=sac_env,
        env=env,
        reward=reward,
        policy=policy,
        state_dim=state_dim,
        action_dim=action_dim,
        fisher_cfg=fisher_cfg,
        device=device,
    )

    inner_optimizer.optimize(
        n_inner_steps,
        inner_loss_fn=lambda x, y, z: inner_loss(x, y, z, outer_optimizer.discount),
        log_every=n_inner_steps // 10,
        n_log_traj=10,
    )

    agent_trajs = collect_trajectories(
        env=env,
        policy=policy,
        n=n_agent_traj,
        max_steps=int(config["env"]["max_steps"]),
        desc="agent outer trajs",
    )

    log_and_checkpoint(outer_step=0, agent_trajs=agent_trajs)

    for outer_step in range(1, n_outer_steps + 1):
        del inner_optimizer

        outer_optimizer.step(expert_train_trajs, agent_trajs)

        policy.load_state_dict(initial_policy_state)
        policy.zero_grad()

        inner_optimizer = make_sac_inner_optimizer(
            sac_env=sac_env,
            env=env,
            reward=reward,
            policy=policy,
            state_dim=state_dim,
            action_dim=action_dim,
            fisher_cfg=fisher_cfg,
            device=device,
        )

        inner_optimizer.optimize(
            n_inner_steps,
            inner_loss_fn=lambda x, y, z: inner_loss(x, y, z, outer_optimizer.discount),
            log_every=n_inner_steps // 10,
            n_log_traj=10,
        )

        agent_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_traj,
            max_steps=int(config["env"]["max_steps"]),
            desc="agent outer trajs",
        )

        log_and_checkpoint(outer_step=outer_step, agent_trajs=agent_trajs)

    sac_env.close()

    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD IRL SAC — Hopper")

    parser.add_argument("--config", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("hopper", args.config)
    config = load_config(config_path)

    fisher_cfg = config["fisher"]
    log_cfg = config["logging"]

    log_dir = log_cfg["log_dir"]
    logger = get_logger("fisher_hopper", log_dir=log_dir)

    set_random_seed(int(fisher_cfg["random_seed"]))
    env = gym.make(config["env"]["id"])
    set_env_seed(env, int(fisher_cfg["env_seed"]))

    try:
        logger.info("=== Fisher-NHD Hopper SAC ===")
        history = train_bilevel(env, config, logger)
        report_path = Path(log_cfg["report_dir"]) / "fisher_sac_hopper_history.json"
        save_history(history, str(report_path))
        logger.info(f"History saved to {report_path}")

    finally:
        print("Terminated")
        env.close()


if __name__ == "__main__":
    main()
