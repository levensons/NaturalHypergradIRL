"""
Fisher-NHD IRL with SAC inner agent for LQR.

Usage:
    python -m src.irl.lqr.fisher
    python -m src.irl.lqr.fisher --config configs/lqr.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch

from src.irl.lqr.env import LQR
from src.irl.lqr.models import Reward, Policy
from src.algorithms.fisher_nhd import FisherNHD
from src.algorithms.sac import SAC
from src.evaluation.metrics import outer_loss, policy_nll, rank_corr, inner_loss, learned_reward_stats
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_length, mean_trajectory_return


def train_fisher(config: dict, logger) -> dict:
    fisher_cfg = config["fisher"]
    inner_cfg = fisher_cfg["inner"]
    sac_cfg = inner_cfg["params"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    env_cfg = config["env"]
    data_cfg = config["data"]
    checkpoint_cfg = config["checkpoint"]

    set_random_seed(int(fisher_cfg["random_seed"]))
    env = LQR(
        A=env_cfg["A"],
        B=env_cfg["B"],
        Q=env_cfg["Q"],
        R=env_cfg["R"],
        process_cov=env_cfg["process_cov"],
        initial_cov=env_cfg["initial_cov"],
        max_episode_steps=int(env_cfg["max_steps"]),
        action_limit=float(env_cfg["action_limit"]),
        observation_limit=env_cfg["observation_limit"],
        termination_state_norm=env_cfg["termination_state_norm"],
        reward_scale=float(env_cfg["reward_scale"]),
        seed=int(fisher_cfg["env_seed"]),
        custom_reward_fn=None,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if inner_cfg["type"] != "sac":
        raise ValueError(f"Expected fisher.inner.type = sac, got {inner_cfg['type']}")

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    random_train_path = Path(data_cfg["random_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    random_train_trajs = load_trajectories(random_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    logger.info(f"Loaded {len(expert_train_trajs)} expert train trajectories from {expert_train_path}")
    logger.info(f"Loaded {len(random_train_trajs)} random train trajectories from {random_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} expert valid trajectories from {expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} random valid trajectories from {random_valid_path}")

    n_outer_steps = int(fisher_cfg["n_outer_steps"])
    n_inner_steps = int(fisher_cfg["n_inner_steps"])

    n_agent_trajs = int(fisher_cfg["n_agent_trajs"])

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
        gamma=float(fisher_cfg["gamma"]),
        alpha=float(fisher_cfg["alpha"]),
        lr=float(fisher_cfg["lr_reward"]),
        fisher_reg=float(fisher_cfg["fisher_reg"]),
        max_grad_norm=fisher_cfg["max_grad_norm"],
        scheduler_gamma=float(fisher_cfg["scheduler_gamma"]),
        sketch_size=int(fisher_cfg["fisher_sketch_size"]),
        fisher_batch_size=int(fisher_cfg["fisher_batch_size"]),
    )

    sac = SAC(
        policy=policy,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=sac_cfg["q_hidden_dim"],
        n_hidden_layers=sac_cfg["q_n_hidden_layers"],
        gamma=fisher_cfg["gamma"],
        alpha=fisher_cfg["alpha"],
        tau=sac_cfg["tau"],
        replay_buffer_capacity=sac_cfg["replay_buffer_capacity"],
    )

    sac_train_env = LQR(
        A=env_cfg["A"],
        B=env_cfg["B"],
        Q=env_cfg["Q"],
        R=env_cfg["R"],
        process_cov=env_cfg["process_cov"],
        initial_cov=env_cfg["initial_cov"],
        max_episode_steps=int(env_cfg["max_steps"]),
        action_limit=float(env_cfg["action_limit"]),
        observation_limit=env_cfg["observation_limit"],
        termination_state_norm=env_cfg["termination_state_norm"],
        reward_scale=float(env_cfg["reward_scale"]),
        seed=int(inner_cfg["train_env_seed"]),
        custom_reward_fn=None,
    )
    sac_eval_env = LQR(
        A=env_cfg["A"],
        B=env_cfg["B"],
        Q=env_cfg["Q"],
        R=env_cfg["R"],
        process_cov=env_cfg["process_cov"],
        initial_cov=env_cfg["initial_cov"],
        max_episode_steps=int(env_cfg["max_steps"]),
        action_limit=float(env_cfg["action_limit"]),
        observation_limit=env_cfg["observation_limit"],
        termination_state_norm=env_cfg["termination_state_norm"],
        reward_scale=float(env_cfg["reward_scale"]),
        seed=int(inner_cfg["eval_env_seed"]),
        custom_reward_fn=None,
    )

    # sac.replay_buffer.extend_from_trajectories(expert_train_trajs)
    sac.replay_buffer.extend_from_trajectories(random_train_trajs)

    def inner_optimize(outer_step: int):
        sac.policy.train()
        
        current_reward_fn = reward.as_fn()
        sac_train_env.custom_reward_fn = current_reward_fn
        sac.replay_buffer.recalc_rewards(current_reward_fn)
        sac.reset_policy()
        sac.reset_critics()
        sac.reset_optimizers()

        def validate(ts: int):
            sac.policy.eval()

            agent_valid_trajs = collect_trajectories(
                env=sac_eval_env,
                policy=sac.policy,
                n=inner_cfg["n_agent_eval_trajs"],
                deterministic=False,
                verbose=False,
            )

            l_inner = inner_loss(sac.policy, reward, agent_valid_trajs, sac.gamma, sac.alpha)

            l_outer = outer_loss(sac.policy, expert_valid_trajs, sac.gamma)

            l_inner_grad = outer_optimizer.d_inner_d_policy(agent_valid_trajs, verbose=False)
            grad_norm = l_inner_grad.norm()
            grad_rms = grad_norm / (l_inner_grad.numel() ** 0.5)
            grad_abs_max = l_inner_grad.abs().max()

            mlflow.log_metrics(
                {
                    f"sac_{outer_step}/l_inner": l_inner,
                    f"sac_{outer_step}/l_outer": l_outer,
                    f"sac_{outer_step}/env_return": float(mean_trajectory_return(agent_valid_trajs)),
                    f"sac_{outer_step}/length": float(mean_trajectory_length(agent_valid_trajs)),
                    f"sac_{outer_step}/learned_return": float(learned_reward_stats(reward, agent_valid_trajs)["return_mean"]),
                    f"sac_{outer_step}/inner_grad_norm": grad_norm.item(),
                    f"sac_{outer_step}/inner_grad_rms": grad_rms.item(),
                    f"sac_{outer_step}/inner_grad_abs_max": grad_abs_max.item(),
                },
                step=ts,
            )

            sac.policy.train()

        sac.optimize(
            train_env=sac_train_env,
            total_steps=n_inner_steps,
            batch_size=int(sac_cfg["batch_size"]),
            max_grad_norm=sac_cfg["max_grad_norm"],
            gradient_update_steps=int(sac_cfg["gradient_update_steps"]),
            target_update_interval=int(sac_cfg["target_update_interval"]),
            critic_lr=float(sac_cfg["critic_lr"]),
            actor_lr=float(sac_cfg["actor_lr"]),
            validate_fn=validate,
            validate_every=int(inner_cfg["validate_every"]),
        )

    ckpt_dir = Path(checkpoint_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = str(ckpt_dir / "fisher.pt")
    best_env_reward = float("-inf")

    mlflow.log_params(
        {
            "gamma": fisher_cfg["gamma"],
            "lr_reward": fisher_cfg["lr_reward"],
            "fisher_reg": fisher_cfg["fisher_reg"],
            "n_outer_steps": fisher_cfg["n_outer_steps"],
            "n_inner_steps": fisher_cfg["n_inner_steps"],
            "n_agent_trajs": fisher_cfg["n_agent_trajs"],
            "reward_hidden": config["reward"]["hidden_dim"],
            "policy_hidden": config["policy"]["hidden_dim"],
            "alpha": fisher_cfg["alpha"],
            "batch_size": sac_cfg["batch_size"],
        }
    )

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

        l_outer = outer_loss(policy, expert_valid_trajs, sac.gamma)

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

    header = (
        f"{'Step':>5} | {'L_outer':>10} | {'agent_len':>10} | "
        f"{'expert_len':>10} | {'agent_ret':>10} | {'expert_ret':>10} | "
        f"{'RankCorr':>9} | {'PolicyNLL':>10} | {'hyp_raw':>10} | "
        f"{'hyp_clip':>10} | {'lr_outer':>12}"
    )
    logger.info(header)

    for outer_step in range(1, n_outer_steps + 1):
        inner_optimize(outer_step)
        agent_train_trajs = collect_trajectories(
            env,
            policy,
            n_agent_trajs,
            deterministic=False,
            desc="agent train trajs",
            verbose=True
        )
        log_and_checkpoint(outer_step, agent_train_trajs)

        if outer_step < n_outer_steps:
            # outer_optimizer.sweep_sketch_sizes(
            #     expert_train_trajs,
            #     agent_train_trajs,
            #     [4, 8, 16, 32, 64, 128, 256, 512],
            #     compare_hypergradients=True
            # )
            outer_optimizer.step(expert_train_trajs, agent_train_trajs)

    env.close()
    sac_train_env.close()
    sac_eval_env.close()
    return history


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-NHD IRL SAC — LQR")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path("lqr", args.config)
    config = load_config(config_path)

    log_cfg = config["logging"]

    log_dir = log_cfg["log_dir"]
    logger = get_logger("fisher_lqr", log_dir=log_dir)

    logger.info("=== Fisher-NHD LQR SAC ===")

    mlflow.set_experiment("fisher")
    with mlflow.start_run(run_name="lqr"):
        history = train_fisher(config, logger)

    report_path = Path(log_cfg["report_dir"]) / "fisher_sac_lqr_history.json"
    save_history(history, str(report_path))
    logger.info(f"History saved to {report_path}")


if __name__ == "__main__":
    main()
