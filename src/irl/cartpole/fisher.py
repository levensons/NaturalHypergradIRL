"""
Fisher-NHD IRL with REINFORCE inner agent for CartPole.

Usage:
    python -m src.irl.cartpole.fisher
    python -m src.irl.cartpole.fisher --config config/cartpole.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch

from src.utils.env import Environment
from src.irl.cartpole.models import Reward, Policy
from src.algorithms.reinforce import REINFORCE
from src.algorithms.fisher_nhd import FisherNHD
from src.evaluation.metrics import inner_loss, learned_reward_stats, outer_loss, policy_nll, rank_corr
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return
from src.evaluation.video import record_policy_video
from src.utils.resources import PeakRAMMonitor


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

    device = torch.device("cpu")
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
        use_sketch=bool(fisher_cfg["use_sketch"]),
        sketch_size=fisher_cfg["fisher_sketch_size"],
        fisher_batch_size=int(fisher_cfg["fisher_batch_size"]),
    )

    reinforce = REINFORCE(
        policy=policy,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
    )

    reinforce_train_env = Environment(
        env_cfg["id"],
        int(inner_cfg["train_env_seed"]),
        int(env_cfg["max_steps"]),
        custom_reward_fn=None,
    )
    reinforce_eval_env = Environment(
        env_cfg["id"],
        int(inner_cfg["eval_env_seed"]),
        int(env_cfg["max_steps"]),
        custom_reward_fn=None,
    )

    def inner_optimize(outer_step: int):
        policy.train()

        current_reward_fn = reward.as_fn()
        reinforce_train_env.custom_reward_fn = current_reward_fn
        reinforce.reset_policy()

        def validate(ts):
            policy.eval()

            agent_valid_trajs = collect_trajectories(
                env=reinforce_eval_env,
                policy=policy,
                n=inner_cfg["n_agent_eval_trajs"],
                deterministic=False,
                verbose=False,
            )

            l_inner = inner_loss(reinforce.policy, reward, agent_valid_trajs, reinforce.gamma, reinforce.alpha)

            l_outer_value = outer_loss(policy, expert_valid_trajs, reinforce.gamma)

            l_inner_grad = outer_optimizer.d_inner_d_policy(agent_valid_trajs, verbose=False)
            grad_norm = l_inner_grad.norm()
            grad_rms = grad_norm / (l_inner_grad.numel() ** 0.5)
            grad_abs_max = l_inner_grad.abs().max()

            mlflow.log_metrics(
                {
                    f"reinforce_{outer_step}/l_inner": float(l_inner),
                    f"reinforce_{outer_step}/l_outer": float(l_outer_value),
                    f"reinforce_{outer_step}/env_return": float(mean_trajectory_return(agent_valid_trajs)),
                    f"reinforce_{outer_step}/length": float(mean_trajectory_length(agent_valid_trajs)),
                    f"reinforce_{outer_step}/learned_return": float(learned_reward_stats(reward, agent_valid_trajs)["return_mean"]),
                    f"reinforce_{outer_step}/inner_grad_norm": grad_norm.item(),
                    f"reinforce_{outer_step}/inner_grad_rms": grad_rms.item(),
                    f"reinforce_{outer_step}/inner_grad_abs_max": grad_abs_max.item(),
                },
                step=ts,
            )

        reinforce.optimize(
            train_env=reinforce_train_env,
            total_steps=n_inner_steps,
            n_traj_per_update=int(reinforce_cfg["n_traj_per_update"]),
            max_grad_norm=reinforce_cfg["max_grad_norm"],
            actor_lr=float(reinforce_cfg["lr_policy"]),
            scheduler_gamma=float(reinforce_cfg["scheduler_gamma"]),
            # validate_fn=validate,
            validate_every=int(inner_cfg["validate_every"]),
        )

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "fisher.pt")
    best_l_outer = float("inf")

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

    mlflow.log_params(
        {
            "env_id": env_cfg["id"],
            "gamma": fisher_cfg["gamma"],
            "alpha": fisher_cfg["alpha"],
            "lr_reward": fisher_cfg["lr_reward"],
            "lr_policy": reinforce_cfg["lr_policy"],
            "fisher_reg": fisher_cfg["fisher_reg"],
            "fisher_sketch_size": fisher_cfg["fisher_sketch_size"],
            "fisher_batch_size": fisher_cfg["fisher_batch_size"],
            "n_outer_steps": n_outer_steps,
            "n_inner_steps": n_inner_steps,
            "n_agent_trajs": n_agent_trajs,
            "n_traj_per_update": reinforce_cfg["n_traj_per_update"],
            "policy_hidden_dim": policy_cfg["hidden_dim"],
            "reward_hidden_dim": reward_cfg["hidden_dim"],
        }
    )
    mlflow.log_params(arch)

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
        nonlocal best_l_outer

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

        if l_outer < best_l_outer:
            best_l_outer = l_outer

            save_checkpoint(
                path=best_checkpoint_path,
                policy=policy,
                reward=reward,
                arch=arch,
                outer_step=outer_step,
                best_l_outer=best_l_outer,
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

    # TRAINING LOOP

    ram_monitor = PeakRAMMonitor(
        output_path="reports/resources/cartpole/fisher.json",
        interval=0.05,
        persist_every=30,
        log_to_mlflow=True,
    )
    ram_monitor.start()

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
            #     [1, 2, 4, 8, 16, 32, 64, 128],
            #     compare_hypergradients=True
            # )
            outer_optimizer.step(expert_train_trajs, agent_train_trajs)

    ram_metrics = ram_monitor.stop()

    mlflow.log_metrics(
        {
            "resources/training_start_rss_mb": ram_metrics["start_rss_mb"],
            "resources/training_peak_rss_mb": ram_metrics["peak_rss_mb"],
            "resources/training_peak_rss_increase_mb": ram_metrics["peak_rss_increase_mb"],
        }
    )

    logger.info(
        "Training memory | "
        f"start RSS={ram_metrics['start_rss_mb']:.2f} MB | "
        f"peak RSS={ram_metrics['peak_rss_mb']:.2f} MB | "
        f"increase={ram_metrics['peak_rss_increase_mb']:.2f} MB"
    )

    env.close()
    reinforce_train_env.close()
    reinforce_eval_env.close()
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
