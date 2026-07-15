"""
Fisher-NHD IRL with SAC inner agent for Hopper.

Usage:
    python -m src.irl.hopper.fisher
    python -m src.irl.hopper.fisher --config configs/hopper.yaml
"""

import argparse
from pathlib import Path
import mlflow

import torch
import torch.nn as nn

from src.algorithms.sac import SAC
from src.algorithms.fisher_nhd import FisherNHD
from src.evaluation.metrics import outer_loss, policy_nll, rank_corr, inner_loss
from src.evaluation.video import record_policy_video
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import Environment
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return


class Reward(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64,
        clamp_magnitude: float = 10.0,
    ):
        super().__init__()

        self.clamp_magnitude = clamp_magnitude

        layers = []
        in_dim = state_dim + action_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([states, actions], dim=-1))
        out = out.squeeze(-1)
        out = torch.clamp(out, -self.clamp_magnitude, self.clamp_magnitude)
        return out # (B,)

    def rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions) # (B,)

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions).sum()
    
    def as_fn(self):
        def reward_fn(states: torch.Tensor, actions: torch.Tensor):
            was_training = self.training
            if was_training:
                self.eval()

            with torch.no_grad():
                out = self.rewards(states, actions)

            if was_training:
                self.train()

            return out # (B,)
        
        return reward_fn


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        hidden_dim: int = 64,
        n_hidden_layers: int = 1,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()

        in_dim = state_dim
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

        self.register_buffer(
            "action_low", torch.as_tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(action_high, dtype=torch.float32)
        )
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, states: torch.Tensor):
        x = self.backbone(states)
        mean = self.mean_head(x)  # (B, action_dim)
        log_std = self.log_std_head(x)  # (B, action_dim)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def action_distribution(self, states: torch.Tensor):
        # states: (B, state_dim)
        # actions: (B, action_dim)
        mean, log_std = self.forward(states)  # (B, action_dim) x 2
        dist = torch.distributions.Normal(mean, torch.exp(log_std))
        return dist

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (B, state_dim)
        # actions: (B, action_dim)
        dist = self.action_distribution(states)

        squashed_actions = (actions - self.action_bias) / self.action_scale # (-1, 1)
        squashed_actions = torch.clamp(squashed_actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)

        raw_actions = 0.5 * (torch.log1p(squashed_actions) - torch.log1p(-squashed_actions))

        log_probs = dist.log_prob(raw_actions)
        correction = torch.log(self.action_scale * (1.0 - torch.pow(squashed_actions, 2)))
        log_probs = log_probs - correction
        return log_probs.sum(dim=-1)  # (B,)

    def sample(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        return_log_probs: bool = False,
    ):
        # states: (B, state_dim)
        dist = self.action_distribution(states)

        raw_actions = dist.mean if deterministic else dist.rsample() # (-inf, +inf)
        squashed_actions = torch.tanh(raw_actions)  # (-1; 1)
        actions = self.action_bias + self.action_scale * squashed_actions # (low, high)
        actions = torch.clamp(actions, min=self.action_low, max=self.action_high)

        if not return_log_probs:
            return actions

        log_probs = dist.log_prob(raw_actions)  # (B, action_dim)
        correction = torch.log(self.action_scale * (1.0 - squashed_actions.pow(2)) + 1e-6)  # (B, action_dim)
        log_probs = log_probs - correction
        log_probs = log_probs.sum(dim=-1)  # (B,)

        return actions, log_probs


def train_bilevel(config: dict, logger) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]
    sac_cfg = inner_cfg["sac"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    env_cfg = config["env"]
    ckpt_cfg = config["checkpoint"]

    set_random_seed(int(fisher_cfg["random_seed"]))
    env = Environment(env_cfg["id"], int(fisher_cfg["env_seed"]), render_mode="rgb_array")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    mlflow.log_params({
        "discount": fisher_cfg["discount"],
        "lr_reward": fisher_cfg["lr_reward"],
        "fisher_reg": fisher_cfg["fisher_reg"],
        "n_outer_steps": fisher_cfg["n_outer_steps"],
        "n_inner_steps": fisher_cfg["n_inner_steps"],
        "n_agent_traj": fisher_cfg["n_agent_traj"],
        "reward_hidden": config["reward"]["hidden_dim"],
        "policy_hidden": config["policy"]["hidden_dim"],
        "alpha": fisher_cfg["alpha"],
        "batch_size": sac_cfg["batch_size"],
    })

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

    hidden_dim = int(policy_cfg["hidden_dim"])
    n_layers = int(policy_cfg["n_hidden_layers"])

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
        action_low=env.action_low,
        action_high=env.action_high,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_layers,
        log_std_min=float(policy_cfg["log_std_min"]),
        log_std_max=float(policy_cfg["log_std_max"]),
    ).to(device)

    outer_optimizer = FisherNHD(
        reward=reward,
        policy=policy,
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        discount=float(fisher_cfg["discount"]),
        alpha=float(fisher_cfg["alpha"]),
        max_grad_norm=float(fisher_cfg["max_grad_norm"]),
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
        sketch_size=int(fisher_cfg["fisher_sketch_size"]),
    )

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
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "policy_hidden": hidden_dim,
        "policy_n_hidden_layers": n_layers,
        "reward_n_hidden_layers": int(reward_cfg["n_hidden_layers"]),
        "reward_hidden": int(reward_cfg["hidden_dim"]),
        "reward_clamp_magnitude": float(reward_cfg["clamp_magnitude"]),
        "log_std_min": float(policy_cfg["log_std_min"]),
        "log_std_max": float(policy_cfg["log_std_max"]),
        "action_low": env.action_low.tolist(),
        "action_high": env.action_high.tolist(),
        "method": "fisher",
        "agent": "sac",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    # def eval_reward_stats():
    #     expert_reward_stats = learned_reward_stats(reward, expert_valid_trajs, outer_optimizer.discount)
    #     random_reward_stats = learned_reward_stats(reward, random_valid_trajs, outer_optimizer.discount)

    #     expert_learned_return = expert_reward_stats["return_mean"]
    #     random_learned_return = random_reward_stats["return_mean"]
    #     expert_random_learned_diff = expert_learned_return - random_learned_return

    #     expert_learned_step_mean = expert_reward_stats["step_mean"]
    #     random_learned_step_mean = random_reward_stats["step_mean"]

    #     logger.info(
    #         f"   [outer] "
    #         f"expert_ret={expert_learned_return:.3f} "
    #         f"random_ret={random_learned_return:.3f} "
    #         f"diff={expert_random_learned_diff:.3f} "
    #         f"expert_step={expert_learned_step_mean:.4f} "
    #         f"random_step={random_learned_step_mean:.4f}"
    #     )

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
                "lr_reward": lr_outer_current,
            },
            step=outer_step,
        )

        record_policy_video(
            env,
            policy,
            video_dir=f"videos/fisher/",
            name_prefix=f"outer_{outer_step}",
            max_steps=1000,
            deterministic=False
        )

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )

    logger.info(header)

    # eval_reward_stats()

    sac = SAC(
        policy=policy,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=sac_cfg["q_hidden_dim"],
        n_hidden_layers=sac_cfg["q_n_hidden_layers"],
        gamma=fisher_cfg["discount"],
        alpha=fisher_cfg["alpha"],
        tau=sac_cfg["tau"],
        replay_buffer_capacity=sac_cfg["replay_buffer_capacity"],
    )

    q1_state = sac.q1.state_dict()
    q2_state = sac.q2.state_dict()

    def inner_optimize(outer_step: int):
        current_reward_fn = reward.as_fn()

        if outer_step > 0:
            sac.replay_buffer.recalc_rewards(current_reward_fn)
            sac.q1.load_state_dict(q1_state)
            sac.q2.load_state_dict(q2_state)
            sac.q1_target.load_state_dict(sac.q1.state_dict())
            sac.q2_target.load_state_dict(sac.q2.state_dict())

        sac_train_env = Environment(id=config["env"]["id"], seed=int(inner_cfg["eval_env_seed"]), custom_reward_fn=current_reward_fn)
        sac_eval_env = Environment(id=config["env"]["id"], seed=int(inner_cfg["train_env_seed"]), custom_reward_fn=None)

        @torch.no_grad()
        def validate(ts: int, n_eval_traj: int = 10):
            sac.policy.eval()

            agent_valid_trajs = collect_trajectories(
                env=sac_eval_env,
                policy=sac.policy,
                n=n_eval_traj,
                max_steps=config["env"]["max_steps"],
                verbose=False,
            )

            l_inner = inner_loss(
                policy=sac.policy,
                reward=reward,
                trajs=agent_valid_trajs,
                discount=sac.gamma,
                alpha=sac.alpha,
            )

            l_outer = outer_loss(
                policy=sac.policy,
                expert_trajs=expert_valid_trajs,
                discount=sac.gamma,
            )

            mlflow.log_metrics(
                {
                    f"sac_{outer_step}/l_inner": float(l_inner),
                    f"sac_{outer_step}/l_outer": float(l_outer),
                },
                step=ts,
            )

            sac.policy.train()

        sac.optimize(
            train_env=sac_train_env,
            total_steps=n_inner_steps,
            learning_starts=int(sac_cfg["learning_starts"]),
            batch_size=int(sac_cfg["batch_size"]),
            max_grad_norm=sac_cfg["max_grad_norm"],
            gradient_update_steps=int(sac_cfg["gradient_update_steps"]),
            target_update_interval=int(sac_cfg["target_update_interval"]),
            critic_lr=float(sac_cfg["critic_lr"]),
            actor_lr=float(sac_cfg["actor_lr"]),
            validate_fn=validate,
            validate_every=1000
        )

        sac_train_env.close()
        sac_eval_env.close()

    inner_optimize(outer_step=0)

    agent_trajs = collect_trajectories(
        env=env,
        policy=policy,
        n=n_agent_traj,
        max_steps=int(config["env"]["max_steps"]),
        desc="agent outer trajs",
    )

    sketch_sweep_results = outer_optimizer.sweep_sketch_sizes(
        expert_trajs=expert_train_trajs,
        agent_trajs=agent_trajs,
        sketch_sizes=[1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512],
        compare_hypergradients=True,
    )

    log_and_checkpoint(outer_step=0, agent_trajs=agent_trajs)

    for outer_step in range(1, n_outer_steps + 1):
        outer_optimizer.step(expert_train_trajs, agent_trajs)
        # eval_reward_stats()

        inner_optimize(outer_step=outer_step)

        agent_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_traj,
            max_steps=int(config["env"]["max_steps"]),
            desc="agent outer trajs",
        )

        log_and_checkpoint(outer_step=outer_step, agent_trajs=agent_trajs)

    env.close()
    
    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD IRL SAC — Hopper")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("hopper", args.config)
    config = load_config(config_path)

    log_cfg = config["logging"]

    log_dir = log_cfg["log_dir"]
    logger = get_logger("fisher_hopper", log_dir=log_dir)

    logger.info("=== Fisher-NHD Hopper SAC ===")

    mlflow.set_experiment("fisher")
    with mlflow.start_run(run_name="hopper"):
        history = train_bilevel(config, logger)

    report_path = Path(log_cfg["report_dir"]) / "fisher_sac_hopper_history.json"
    save_history(history, str(report_path))
    logger.info(f"History saved to {report_path}")


if __name__ == "__main__":
    main()
