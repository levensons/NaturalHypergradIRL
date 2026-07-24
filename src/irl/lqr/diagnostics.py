"""Post-run diagnostics for the neural Fisher-NHD LQR benchmark.

The diagnostic freezes a learned reward from a Fisher checkpoint, compares the
ground-truth quadratic features of expert and checkpoint-policy trajectories,
and then gives SAC an extended optimization budget under that fixed reward.

Usage::

    python -m src.irl.lqr.diagnostics --config config/neural_lqr.yaml
    python -m src.irl.lqr.diagnostics --checkpoint path/to/checkpoint.pt \
        --total-steps 30000
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from src.evaluation.metrics import inner_loss, learned_reward_stats, policy_entropy
from src.irl.hopper.fisher import Reward
from src.irl.lqr.env import register_lqr_env
from src.irl.lqr.sac_oracle import build_policy_and_sac
from src.utils.config import load_config
from src.utils.data import load_trajectories
from src.utils.env import Environment
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories, mean_trajectory_return


_OUTER_STEP_PATTERN = re.compile(r"_outer_(\d+)\.pt$")


def resolve_fisher_checkpoint(config: dict, checkpoint: str | None = None) -> Path:
    """Resolve an explicit checkpoint or the latest outer-step checkpoint."""
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Fisher checkpoint does not exist: {path}")
        return path

    checkpoint_dir = Path(config["checkpoint"]["dir"])
    candidates = list(checkpoint_dir.glob("fisher_sac_*_outer_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No Fisher checkpoints found under {checkpoint_dir}. Pass --checkpoint."
        )

    def sort_key(path: Path) -> tuple[int, int]:
        match = _OUTER_STEP_PATTERN.search(path.name)
        outer_step = int(match.group(1)) if match else -1
        return outer_step, path.stat().st_mtime_ns

    return max(candidates, key=sort_key)


def quadratic_feature_stats(
    trajectories,
    *,
    Q,
    R,
    reward_scale: float = 1.0,
) -> dict:
    """Summarize true LQR features and costs from state-action trajectories."""
    if not trajectories:
        raise ValueError("At least one trajectory is required.")

    q = torch.as_tensor(Q, dtype=torch.float64)
    r = torch.as_tensor(R, dtype=torch.float64)
    state_dim = q.shape[0]
    action_dim = r.shape[0]
    if q.shape != (state_dim, state_dim):
        raise ValueError(f"Q must be square, got {tuple(q.shape)}.")
    if r.shape != (action_dim, action_dim):
        raise ValueError(f"R must be square, got {tuple(r.shape)}.")

    state_squares = []
    action_squares = []
    state_square_sums = []
    action_square_sums = []
    state_cost_sums = []
    action_cost_sums = []
    lengths = []

    for trajectory in trajectories:
        states = torch.as_tensor(trajectory["states"], dtype=torch.float64)
        actions = torch.as_tensor(trajectory["actions"], dtype=torch.float64)
        if states.ndim != 2 or states.shape[1] != state_dim:
            raise ValueError(
                f"Trajectory states must have shape (T, {state_dim}), "
                f"got {tuple(states.shape)}."
            )
        if actions.ndim != 2 or actions.shape[1] != action_dim:
            raise ValueError(
                f"Trajectory actions must have shape (T, {action_dim}), "
                f"got {tuple(actions.shape)}."
            )
        if states.shape[0] != actions.shape[0] or states.shape[0] == 0:
            raise ValueError("Trajectory states/actions must have the same positive length.")

        squared_states = states.square()
        squared_actions = actions.square()
        state_squares.append(squared_states)
        action_squares.append(squared_actions)
        state_square_sums.append(squared_states.sum(dim=0))
        action_square_sums.append(squared_actions.sum(dim=0))
        state_cost_sums.append(torch.einsum("ti,ij,tj->", states, q, states))
        action_cost_sums.append(torch.einsum("ti,ij,tj->", actions, r, actions))
        lengths.append(states.shape[0])

    all_state_squares = torch.cat(state_squares, dim=0)
    all_action_squares = torch.cat(action_squares, dim=0)
    mean_state_square_sums = torch.stack(state_square_sums).mean(dim=0)
    mean_action_square_sums = torch.stack(action_square_sums).mean(dim=0)
    state_cost_sum_mean = torch.stack(state_cost_sums).mean().item()
    action_cost_sum_mean = torch.stack(action_cost_sums).mean().item()
    total_cost_sum_mean = state_cost_sum_mean + action_cost_sum_mean

    return {
        "n_trajectories": len(trajectories),
        "n_steps": int(sum(lengths)),
        "length_mean": float(np.mean(lengths)),
        "per_step_mean": {
            **{
                f"x{index + 1}_squared": value.item()
                for index, value in enumerate(all_state_squares.mean(dim=0))
            },
            **{
                f"u{index + 1}_squared": value.item()
                for index, value in enumerate(all_action_squares.mean(dim=0))
            },
        },
        "trajectory_sum_mean": {
            **{
                f"x{index + 1}_squared": value.item()
                for index, value in enumerate(mean_state_square_sums)
            },
            **{
                f"u{index + 1}_squared": value.item()
                for index, value in enumerate(mean_action_square_sums)
            },
        },
        "state_cost_sum_mean": state_cost_sum_mean,
        "action_cost_sum_mean": action_cost_sum_mean,
        "total_cost_sum_mean": total_cost_sum_mean,
        "implied_env_return_mean": -0.5 * float(reward_scale) * total_cost_sum_mean,
        "observed_env_return_mean": mean_trajectory_return(trajectories),
    }


def feature_ratios(candidate: dict, expert: dict) -> dict[str, float]:
    """Return candidate/expert ratios for the interpretable squared features."""
    ratios = {}
    for name, expert_value in expert["trajectory_sum_mean"].items():
        candidate_value = candidate["trajectory_sum_mean"][name]
        ratios[name] = (
            float(candidate_value / expert_value)
            if expert_value != 0.0
            else float("nan")
        )
    return ratios


def _collect_policy_trajectories(
    config: dict,
    policy,
    *,
    n_trajectories: int,
    seed: int,
):
    device = next(policy.parameters()).device
    fork_devices = [device] if device.type == "cuda" else []
    # Evaluation uses stochastic SAC actions. Isolate its random stream so that
    # validation neither perturbs training nor changes when repeated.
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        env = Environment(config["env"]["id"], seed=seed)
        try:
            return collect_trajectories(
                env=env,
                policy=policy,
                n=n_trajectories,
                max_steps=int(config["env"]["max_steps"]),
                verbose=False,
            )
        finally:
            env.close()


def _policy_summary(config: dict, policy, reward, trajectories) -> dict:
    fisher_cfg = config["fisher"]
    lqr_cfg = config["lqr"]
    return {
        "env_return_mean": mean_trajectory_return(trajectories),
        "inner_loss": inner_loss(
            policy=policy,
            reward=reward,
            trajs=trajectories,
            discount=float(fisher_cfg["discount"]),
            alpha=float(fisher_cfg["alpha"]),
        ),
        "policy_entropy": policy_entropy(policy, trajectories),
        "learned_reward": learned_reward_stats(
            reward, trajectories, discount=float(fisher_cfg["discount"])
        ),
        "quadratic_features": quadratic_feature_stats(
            trajectories,
            Q=lqr_cfg["Q"],
            R=lqr_cfg["R"],
            reward_scale=float(lqr_cfg["reward_scale"]),
        ),
    }


def _clone_state_dict(module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def run_diagnostics(
    config: dict,
    logger,
    *,
    checkpoint_path: Path,
    total_steps: int,
    n_eval_traj: int,
    validate_every: int,
    policy_init: str,
) -> tuple[dict, dict]:
    """Run feature comparison and fixed-learned-reward SAC optimization."""
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative.")
    if n_eval_traj <= 0 or validate_every <= 0:
        raise ValueError("n_eval_traj and validate_every must be positive.")
    if policy_init not in {"checkpoint", "random"}:
        raise ValueError("policy_init must be 'checkpoint' or 'random'.")

    register_lqr_env(config)
    diagnostics_cfg = config.get("diagnostics", {})
    random_seed = int(diagnostics_cfg.get("random_seed", config["fisher"]["random_seed"]))
    eval_seed = int(diagnostics_cfg.get("eval_env_seed", 300))
    train_seed = int(diagnostics_cfg.get("train_env_seed", 301))
    set_random_seed(random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy, sac, (state_dim, action_dim) = build_policy_and_sac(config, device)
    random_policy_state = _clone_state_dict(policy)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    checkpoint_policy = copy.deepcopy(policy).eval()
    if policy_init == "random":
        policy.load_state_dict(random_policy_state)

    reward_cfg = config["reward"]
    reward = Reward(
        state_dim=state_dim,
        action_dim=action_dim,
        n_hidden_layers=int(reward_cfg["n_hidden_layers"]),
        hidden_dim=int(reward_cfg["hidden_dim"]),
        clamp_magnitude=float(reward_cfg["clamp_magnitude"]),
    ).to(device)
    reward.load_state_dict(checkpoint["reward_state_dict"])
    reward.eval()
    reward.requires_grad_(False)

    expert_trajectories = load_trajectories(
        Path(config["data"]["expert_valid_trajs"]), map_location="cpu"
    )
    checkpoint_trajectories = _collect_policy_trajectories(
        config,
        checkpoint_policy,
        n_trajectories=n_eval_traj,
        seed=eval_seed,
    )
    expert_features = quadratic_feature_stats(
        expert_trajectories,
        Q=config["lqr"]["Q"],
        R=config["lqr"]["R"],
        reward_scale=float(config["lqr"]["reward_scale"]),
    )
    expert_learned_reward = learned_reward_stats(
        reward,
        expert_trajectories,
        discount=float(config["fisher"]["discount"]),
    )
    checkpoint_summary = _policy_summary(
        config, checkpoint_policy, reward, checkpoint_trajectories
    )
    checkpoint_ratios = feature_ratios(
        checkpoint_summary["quadratic_features"], expert_features
    )
    ratio_text = " | ".join(
        f"{name}_ratio={value:.3f}"
        for name, value in checkpoint_ratios.items()
    )
    logger.info(
        "Feature baseline | "
        f"expert_return={expert_features['observed_env_return_mean']:.3f} | "
        f"checkpoint_return={checkpoint_summary['env_return_mean']:.3f} | "
        f"{ratio_text}"
    )

    history = {"validations": []}
    best_inner_loss = float("inf")
    best_policy_state = None

    def validate(ts: int) -> None:
        nonlocal best_inner_loss, best_policy_state
        trajectories = _collect_policy_trajectories(
            config,
            sac.policy,
            n_trajectories=n_eval_traj,
            seed=eval_seed,
        )
        summary = _policy_summary(config, sac.policy, reward, trajectories)
        record = {
            "phase_step": int(ts),
            "env_step": int(sac.total_env_steps),
            **summary,
        }
        history["validations"].append(record)
        if summary["inner_loss"] < best_inner_loss:
            best_inner_loss = float(summary["inner_loss"])
            best_policy_state = _clone_state_dict(sac.policy)
        features = summary["quadratic_features"]["trajectory_sum_mean"]
        feature_text = " | ".join(
            f"{name}={value:.3f}" for name, value in features.items()
        )
        logger.info(
            "Fixed-reward SAC validation | "
            f"env_steps={sac.total_env_steps} | "
            f"inner_loss={summary['inner_loss']:.3f} | "
            f"learned_return={summary['learned_reward']['return_mean']:.3f} | "
            f"true_return={summary['env_return_mean']:.3f} | "
            f"{feature_text}"
        )

    training_started = perf_counter()
    train_env = Environment(config["env"]["id"], seed=train_seed)
    sac_cfg = config["fisher"]["inner"]["sac"]
    try:
        validate(0)
        if total_steps > 0:
            sac.optimize(
                train_env=train_env,
                total_steps=total_steps,
                learning_starts=int(sac_cfg["learning_starts"]),
                batch_size=int(sac_cfg["batch_size"]),
                max_grad_norm=sac_cfg["max_grad_norm"],
                gradient_update_steps=int(sac_cfg["gradient_update_steps"]),
                target_update_interval=int(sac_cfg["target_update_interval"]),
                critic_lr=float(sac_cfg["critic_lr"]),
                actor_lr=float(sac_cfg["actor_lr"]),
                reward_fn=reward.rewards,
                validate_fn=validate,
                validate_every=validate_every,
            )
            if history["validations"][-1]["env_step"] != sac.total_env_steps:
                validate(total_steps)
    finally:
        train_env.close()
    training_seconds = perf_counter() - training_started

    final_policy_state = _clone_state_dict(sac.policy)
    final_summary = history["validations"][-1]
    if best_policy_state is None:
        raise RuntimeError("Fixed-reward SAC validation did not produce a policy.")
    sac.policy.load_state_dict(best_policy_state)
    best_trajectories = _collect_policy_trajectories(
        config, sac.policy, n_trajectories=n_eval_traj, seed=eval_seed
    )
    best_summary = _policy_summary(config, sac.policy, reward, best_trajectories)

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_outer_step": checkpoint.get("outer_step"),
        "device": str(device),
        "policy_initialization": policy_init,
        "critic_initialization": "random",
        "replay_initialization": "empty",
        "total_steps": total_steps,
        "validate_every": validate_every,
        "n_eval_trajectories": n_eval_traj,
        "training_seconds": training_seconds,
        "expert_quadratic_features": expert_features,
        "expert_learned_reward": expert_learned_reward,
        "checkpoint_policy": {
            **checkpoint_summary,
            "feature_ratio_to_expert": checkpoint_ratios,
        },
        "fixed_reward_sac_final": {
            **final_summary,
            "feature_ratio_to_expert": feature_ratios(
                final_summary["quadratic_features"], expert_features
            ),
        },
        "fixed_reward_sac_best_inner_loss": {
            **best_summary,
            "feature_ratio_to_expert": feature_ratios(
                best_summary["quadratic_features"], expert_features
            ),
        },
        "history": history,
    }
    policy_checkpoint = {
        "source_fisher_checkpoint": str(checkpoint_path),
        "policy_initialization": policy_init,
        "total_env_steps": sac.total_env_steps,
        "best_inner_loss": best_inner_loss,
        "best_policy_state_dict": best_policy_state,
        "final_policy_state_dict": final_policy_state,
    }
    return report, policy_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neural_lqr.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-eval-traj", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument(
        "--policy-init", choices=("checkpoint", "random"), default=None
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    diagnostics_cfg = config.get("diagnostics", {})
    checkpoint_path = resolve_fisher_checkpoint(config, args.checkpoint)
    total_steps = int(
        args.total_steps
        if args.total_steps is not None
        else diagnostics_cfg.get("total_steps", 30000)
    )
    n_eval_traj = int(
        args.n_eval_traj
        if args.n_eval_traj is not None
        else diagnostics_cfg.get("n_eval_traj", 50)
    )
    validate_every = int(
        args.validate_every
        if args.validate_every is not None
        else diagnostics_cfg.get("validate_every", 1000)
    )
    policy_init = str(
        args.policy_init
        if args.policy_init is not None
        else diagnostics_cfg.get("policy_init", "checkpoint")
    )

    logger = get_logger(
        f"{config['env']['name']}_diagnostics",
        log_dir=config["logging"]["log_dir"],
    )
    logger.info(f"Using Fisher checkpoint: {checkpoint_path}")
    report, policy_checkpoint = run_diagnostics(
        config,
        logger,
        checkpoint_path=checkpoint_path,
        total_steps=total_steps,
        n_eval_traj=n_eval_traj,
        validate_every=validate_every,
        policy_init=policy_init,
    )

    report_path = Path(
        diagnostics_cfg.get(
            "report_path",
            Path(config["logging"]["report_dir"]) / "final_reward_diagnostics.json",
        )
    )
    save_history(report, str(report_path))
    policy_path = Path(
        diagnostics_cfg.get(
            "policy_path",
            Path(config["checkpoint"]["dir"]) / "final_reward_sac_diagnostic.pt",
        )
    )
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_checkpoint, policy_path)
    logger.info(
        f"Saved diagnostics to {report_path} and diagnostic policy to {policy_path}"
    )


if __name__ == "__main__":
    main()
