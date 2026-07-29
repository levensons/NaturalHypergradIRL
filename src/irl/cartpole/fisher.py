"""
Fisher-NHD IRL with REINFORCE inner agent for CartPole.

Usage:
    python -m src.irl.cartpole.fisher
    python -m src.irl.cartpole.fisher --config configs/cartpole.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.algorithms.reinforce import REINFORCE
from src.algorithms.fisher_nhd import FisherNHD
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
from src.utils.env import Environment
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed
from src.utils.trajectories import (
    collect_trajectories,
    mean_trajectory_length,
    mean_trajectory_return,
)
from src.evaluation.video import record_policy_video


class Reward(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64,
        clamp_magnitude: float = 10.0,
    ):
        super().__init__()

        self.action_dim = int(action_dim)
        self.clamp_magnitude = float(clamp_magnitude)

        layers = []
        in_dim = state_dim + action_dim

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def _prepare_actions(self, actions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        actions = torch.as_tensor(actions, device=actions.device)

        if actions.ndim > 0 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        actions = actions.long()
        return F.one_hot(actions, num_classes=self.action_dim).to(dtype=dtype)

    def forward(self,states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        actions_one_hot = self._prepare_actions(actions, states.dtype)
        out = torch.cat([states, actions_one_hot], dim=-1)
        out = self.net(out)
        out = torch.clamp(out, -self.clamp_magnitude, self.clamp_magnitude)
        out = out.squeeze(-1)
        return out  # (B,)

    def rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions)

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions).sum()

    def as_fn(self):
        def reward_fn(states: torch.Tensor, actions: torch.Tensor):
            device = next(self.parameters()).device
            was_training = self.training

            if was_training:
                self.eval()

            with torch.no_grad():
                states_device = torch.as_tensor(
                    states,
                    dtype=torch.float32,
                    device=device,
                )
                actions_device = torch.as_tensor(actions, device=device)
                out = self.rewards(states_device, actions_device)
                return out

            if was_training:
                self.train()

        return reward_fn


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 1,
    ):
        super().__init__()

        layers = []
        in_dim = state_dim

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def action_distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(states))

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim > 0 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        dist = self.action_distribution(states)
        log_probs = dist.log_prob(actions.long())
        return log_probs

    def sample(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        return_log_probs: bool = False,
    ):
        dist = self.action_distribution(states)
        actions = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()

        if not return_log_probs:
            return actions

        return actions, dist.log_prob(actions)


def train_fisher(config: dict, logger) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]
    reinforce_cfg = inner_cfg["params"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    env_cfg = config["env"]
    data_cfg = config["data"]
    ckpt_cfg = config["checkpoint"]

    if inner_cfg["type"] != "reinforce":
        raise ValueError("Expected fisher.inner.type = reinforce, " f"got {inner_cfg['type']}.")

    set_random_seed(int(fisher_cfg["random_seed"]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    env = Environment(
        id=env_cfg["id"],
        seed=int(fisher_cfg["env_seed"]),
        max_episode_steps=int(env_cfg["max_steps"]),
        render_mode="rgb_array",
    )

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    random_train_path = Path(data_cfg["random_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    random_train_trajs = load_trajectories(random_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    logger.info(f"Loaded {len(expert_train_trajs)} expert train trajectories " f"from {expert_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} expert valid trajectories " f"from {expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} random valid trajectories " f"from {random_valid_path}")

    n_outer_steps = int(fisher_cfg["n_outer_steps"])
    n_inner_steps = int(fisher_cfg["n_inner_steps"])
    n_agent_trajs = int(fisher_cfg["n_agent_trajs"])

    reward = Reward(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_hidden_layers=int(reward_cfg["n_hidden_layers"]),
        hidden_dim=int(reward_cfg["hidden_dim"]),
        clamp_magnitude=float(reward_cfg["clamp_magnitude"]),
    ).to(device)

    policy = Policy(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(policy_cfg["hidden_dim"]),
        n_hidden_layers=int(policy_cfg["n_hidden_layers"]),
    ).to(device)

    outer_optimizer = FisherNHD(
        reward=reward,
        policy=policy,
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
        max_grad_norm=fisher_cfg["max_grad_norm"],
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
        sketch_size=int(fisher_cfg["fisher_sketch_size"]),
    )

    reinforce = REINFORCE(
        policy=policy,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
    )

    train_env = Environment(env_cfg["id"], int(inner_cfg["train_env_seed"]), int(env_cfg["max_steps"]), custom_reward_fn=None)
    eval_env = Environment(env_cfg["id"], int(inner_cfg["eval_env_seed"]), int(env_cfg["max_steps"]), custom_reward_fn=None)

    policy_state = torch.nn.utils.parameters_to_vector(policy.parameters()).clone().detach()

    def inner_optimize(outer_step: int):
        policy.train()

        current_reward_fn = reward.as_fn()
        train_env.custom_reward_fn = current_reward_fn

        torch.nn.utils.vector_to_parameters(policy_state.clone().detach(), policy.parameters())
        reinforce.reset_policy_optimizer()

        def validate(ts):
            policy.eval()

            agent_eval_trajs = collect_trajectories(
                env=eval_env,
                policy=policy,
                n=inner_cfg["n_agent_eval_trajs"],
                deterministic=False,
                verbose=False,
            )

            l_inner = inner_loss(
                policy=policy,
                reward=reward,
                trajs=agent_eval_trajs,
                gamma=reinforce.gamma,
                alpha=reinforce.alpha,
            )
            l_outer_value = outer_loss(
                policy=policy,
                expert_trajs=expert_valid_trajs,
                gamma=reinforce.gamma,
            )

            inner_grad = outer_optimizer.d_inner_d_policy(
                agent_eval_trajs,
                verbose=False,
            )

            grad_norm = inner_grad.norm()
            grad_rms = grad_norm / inner_grad.numel() ** 0.5
            grad_abs_max = inner_grad.abs().max()
            policy_vector = torch.nn.utils.parameters_to_vector(policy.parameters()).detach()

            grad_relative = (grad_norm / policy_vector.norm().clamp_min(1e-12))

            with torch.no_grad():
                all_states = torch.cat([traj["states"].to(device=device, dtype=torch.float32) for traj in agent_eval_trajs], dim=0)

                dist = policy.action_distribution(all_states)
                probs = dist.probs  # (N_states, action_dim)

                policy_entropy = dist.entropy().mean()
                max_action_prob = probs.max(dim=-1).values.mean()

                deterministic_fraction_95 = (probs.max(dim=-1).values > 0.95).float().mean()

                deterministic_fraction_99 = (probs.max(dim=-1).values > 0.99).float().mean()

                action_0_prob = probs[:, 0].mean()
                action_1_prob = probs[:, 1].mean()

            mlflow.log_metrics(
                {
                    f"reinforce_{outer_step}/l_inner": float(l_inner),
                    f"reinforce_{outer_step}/l_outer": float(l_outer_value),
                    f"reinforce_{outer_step}/env_return": float(mean_trajectory_return(agent_eval_trajs)),
                    f"reinforce_{outer_step}/length": float(mean_trajectory_length(agent_eval_trajs)),
                    f"reinforce_{outer_step}/inner_grad_norm": grad_norm.item(),
                    f"reinforce_{outer_step}/inner_grad_rms": grad_rms.item(),
                    f"reinforce_{outer_step}/inner_grad_abs_max": grad_abs_max.item(),
                    f"reinforce_{outer_step}/inner_grad_relative": grad_relative.item(),
                    f"reinforce_{outer_step}/policy_entropy": policy_entropy.item(),
                    f"reinforce_{outer_step}/max_action_prob": max_action_prob.item(),
                    f"reinforce_{outer_step}/deterministic_fraction_95": deterministic_fraction_95.item(),
                    f"reinforce_{outer_step}/deterministic_fraction_99": deterministic_fraction_99.item(),
                    f"reinforce_{outer_step}/action_0_prob": action_0_prob.item(),
                    f"reinforce_{outer_step}/action_1_prob": action_1_prob.item(),
                },
                step=ts,
            )

        reinforce.optimize(
            train_env=train_env,
            total_steps=n_inner_steps,
            n_traj_per_update=int(reinforce_cfg["n_traj_per_update"]),
            max_grad_norm=reinforce_cfg["max_grad_norm"],
            actor_lr=float(reinforce_cfg["lr_policy"]),
            scheduler_gamma=float(reinforce_cfg["scheduler_gamma"]),
            validate_fn=validate,
            validate_every=int(inner_cfg["validate_every"]),
        )

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "fisher.pt")
    best_env_reward = float("-inf")

    mlflow.log_params(
        {
            "env_id": env_cfg["id"],
            "gamma": fisher_cfg["gamma"],
            "alpha": fisher_cfg["alpha"],
            "lr_reward": fisher_cfg["lr_reward"],
            "lr_policy": reinforce_cfg["lr_policy"],
            "fisher_reg": fisher_cfg["fisher_reg"],
            "use_sketch": fisher_cfg.get("use_sketch", True),
            "sketch_size": fisher_cfg.get("sketch_size", 64),
            "n_outer_steps": n_outer_steps,
            "n_inner_steps": n_inner_steps,
            "n_agent_trajs": n_agent_trajs,
            "n_traj_per_update": reinforce_cfg["n_traj_per_update"],
            "policy_hidden_dim": policy_cfg["hidden_dim"],
            "reward_hidden_dim": reward_cfg["hidden_dim"],
        }
    )

    arch = {
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "policy_hidden": int(policy_cfg["hidden_dim"]),
        "policy_n_hidden_layers": int(policy_cfg["n_hidden_layers"]),
        "reward_hidden_dim": int(reward_cfg["hidden_dim"]),
        "reward_n_hidden_layers": int(reward_cfg["n_hidden_layers"]),
        "reward_clamp_magnitude": float(reward_cfg["clamp_magnitude"]),
        "method": "fisher",
        "agent": "reinforce",
        "env_name": env_cfg["name"],
        "env_id": env_cfg["id"],
        "action_type": env_cfg["action_type"],
    }

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
        "expert_learned_return": [],
        "random_learned_return": [],
        "expert_learned_step_mean": [],
        "random_learned_step_mean": [],
    }

    def log_and_checkpoint(outer_step: int, agent_trajs):
        nonlocal best_env_reward

        lr_outer_current = outer_optimizer.optimizer.param_groups[0]["lr"]
        raw_hypgrad_norm = outer_optimizer.raw_grad_norm
        clipped_hypgrad_norm = outer_optimizer.clipped_grad_norm

        l_outer = outer_loss(policy, expert_valid_trajs, reinforce.gamma)

        agent_len = mean_trajectory_length(agent_trajs)
        expert_len = mean_trajectory_length(expert_train_trajs)

        agent_ret = mean_trajectory_return(agent_trajs)
        expert_ret = mean_trajectory_return(expert_train_trajs)

        rank_corr_val = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
        policy_nll_val = policy_nll(policy, expert_valid_trajs)

        expert_reward_stats = learned_reward_stats(reward, expert_valid_trajs)
        random_reward_stats = learned_reward_stats(reward, random_valid_trajs)
        expert_learned_return_valid = float(expert_reward_stats["return_mean"])
        random_learned_return_valid = float(random_reward_stats["return_mean"])
        expert_step_mean = float(expert_reward_stats["step_mean"])
        random_step_mean = float(random_reward_stats["step_mean"])

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
        history["expert_learned_return"].append(expert_learned_return_valid)
        history["random_learned_return"].append(random_learned_return_valid)
        history["expert_learned_step_mean"].append(expert_step_mean)
        history["random_learned_step_mean"].append(random_step_mean)

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

        logger.info(
            f"{outer_step:>5} | {l_outer:>10.3f} | {agent_len:>10.1f} | "
            f"{expert_len:>10.1f} | {agent_ret:>10.1f} | {expert_ret:>10.1f} | "
            f"{rank_corr_val:>9.3f} | {policy_nll_val:>10.3f} | "
            f"{raw_hypgrad_norm:>10.3f} | {clipped_hypgrad_norm:>10.3f} | "
            f"{lr_outer_current:>12.2e}"
        )

        logger.info(
            "   [reward] "
            f"expert_ret={expert_learned_return_valid:.3f} "
            f"random_ret={random_learned_return_valid:.3f} "
            f"expert_step={expert_step_mean:.4f} "
            f"random_step={random_step_mean:.4f} "
        )

        mlflow.log_metrics(
            {
                "outer_loss": l_outer,
                "agent_return": agent_ret,
                "expert_return": expert_ret,
                "agent_length": agent_len,
                "rank_corr": rank_corr_val,
                "policy_nll": policy_nll_val,
                "hypergrad_norm": raw_hypgrad_norm,
                "hypergrad_norm_clipped": clipped_hypgrad_norm,
                "lr_outer": lr_outer_current,
                "reward/expert_return": expert_learned_return_valid,
                "reward/random_return": random_learned_return_valid,
                "reward/expert_step_mean": expert_step_mean,
                "reward/random_step_mean": random_step_mean,
            },
            step=outer_step,
        )

        # record_policy_video(
        #     env,
        #     policy,
        #     video_dir=f"videos/cartpole/fisher/",
        #     name_prefix=f"outer_{outer_step}",
        #     deterministic=False,
        #     device=device
        # )

    logger.info(
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )

    for outer_step in range(1, n_outer_steps + 1):
        inner_optimize(outer_step)
        agent_train_trajs = collect_trajectories(
            env,
            policy,
            n_agent_trajs,
            deterministic=False,
            desc="agent outer trajs",
            verbose=True,
        )
        log_and_checkpoint(outer_step, agent_train_trajs)

        if outer_step < n_outer_steps:
            # outer_optimizer.sweep_sketch_sizes(
            #     expert_train_trajs,
            #     agent_train_trajs,
            #     [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            #     compare_hypergradients=True
            # )
            # outer_optimizer.sweep_n_agent_trajs(
            #     expert_train_trajs,
            #     agent_train_trajs,
            #     [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            #     verbose=True
            # )
            # outer_optimizer.compare_agent_trajectory_sets(
            #     expert_train_trajs,
            #     agent_train_trajs[:n_agent_trajs//2],
            #     agent_train_trajs[n_agent_trajs//2:],
            #     verbose=True
            # )
            outer_optimizer.step(expert_train_trajs, agent_train_trajs)

    env.close()
    train_env.close()
    eval_env.close()
    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD IRL REINFORCE — CartPole")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("cartpole", args.config)
    config = load_config(config_path)

    log_cfg = config["logging"]
    logger = get_logger("fisher_cartpole", log_dir=log_cfg["log_dir"])
    logger.info("=== Fisher-NHD CartPole REINFORCE ===")

    mlflow.set_experiment("fisher")
    with mlflow.start_run(run_name="cartpole"):
        history = train_fisher(config, logger)

    report_path = Path(log_cfg["report_dir"]) / "fisher_reinforce_cartpole_history.json"
    save_history(history, str(report_path))
    logger.info(f"History saved to {report_path}")


if __name__ == "__main__":
    main()
