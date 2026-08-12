"""
Evaluate a saved IRL checkpoint.

Usage:
    python -m src.evaluation.evaluate \
        --env lqr \
        --config config/lqr.yaml \
        --checkpoint checkpoints/lqr/fisher/exp1.pt \
        --report-dir reports/metrics/lqr/exp1

Static check only:
    python -m src.evaluation.evaluate \
        --env lqr \
        --config config/lqr.yaml \
        --checkpoint checkpoints/lqr/fisher/exp1.pt \
        --check-only
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.evaluation.bootstrap import bootstrap_metric, bootstrap_two_group_metric
from src.evaluation.metrics import env_reward, policy_nll, rank_corr
from src.irl.builders import SUPPORTED_AGENTS, SUPPORTED_ENVS, import_env_builders
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.data import load_trajectories
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories


SUPPORTED_METHODS = {"fisher", "ml_irl"}


def resolve_method_agent(checkpoint: dict) -> tuple[str, str]:
    arch = checkpoint["arch"]

    method = arch.get("method")
    agent = arch.get("agent")

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported or missing IRL method in checkpoint: {method}.")

    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported or missing agent in checkpoint: {agent}.")

    return method, agent


def compute_bootstrap_metrics(
    policy,
    reward,
    agent_test_trajs: list,
    expert_test_trajs: list,
    random_test_trajs: list,
    bootstrap_cfg: dict,
) -> dict[str, dict[str, float]] | None:
    if not bool(bootstrap_cfg.get("enabled", False)):
        return None

    n_samples = int(bootstrap_cfg["n_samples"])
    seed = int(bootstrap_cfg["seed"])

    return {
        "PolicyNLL": bootstrap_metric(
            metric_fn=lambda trajectories: policy_nll(policy, trajectories),
            trajectories=expert_test_trajs,
            n_samples=n_samples,
            seed=seed + 1,
            desc="Bootstrap PolicyNLL",
        ),
        "EnvReward": bootstrap_metric(
            metric_fn=env_reward,
            trajectories=agent_test_trajs,
            n_samples=n_samples,
            seed=seed + 2,
            desc="Bootstrap EnvReward",
        ),
        "ExpertRet": bootstrap_metric(
            metric_fn=env_reward,
            trajectories=expert_test_trajs,
            n_samples=n_samples,
            seed=seed + 3,
            desc="Bootstrap ExpertRet",
        ),
        "RandomRet": bootstrap_metric(
            metric_fn=env_reward,
            trajectories=random_test_trajs,
            n_samples=n_samples,
            seed=seed + 4,
            desc="Bootstrap RandomRet",
        ),
        "RankCorr": bootstrap_two_group_metric(
            metric_fn=lambda expert, random: rank_corr(reward, expert + random),
            first=expert_test_trajs,
            second=random_test_trajs,
            n_samples=n_samples,
            seed=seed + 5,
            desc="Bootstrap RankCorr",
        ),
    }


def metric_result(
    value: float,
    bootstrap_metrics: dict[str, dict[str, float]] | None,
    name: str,
) -> dict:
    result = {"value": float(value)}

    if bootstrap_metrics is not None and name in bootstrap_metrics:
        result["bootstrap"] = bootstrap_metrics[name]

    return result


def format_metric(
    name: str,
    value: float,
    bootstrap_metrics: dict[str, dict[str, float]] | None,
    digits: int = 4,
) -> str:
    if bootstrap_metrics is None or name not in bootstrap_metrics:
        return f"{name:<10} = {value:.{digits}f}"

    stats = bootstrap_metrics[name]

    return (
        f"{name:<10} = {value:.{digits}f} "
        f"± {stats['std']:.{digits}f} "
        f"(boot_mean={stats['mean']:.{digits}f})"
    )


def load_test_trajectories(config: dict) -> tuple[list, list]:
    data_cfg = config["data"]

    expert_test_path = Path(data_cfg["expert_test_trajs"])
    random_test_path = Path(data_cfg["random_test_trajs"])

    expert_test_trajs = load_trajectories(expert_test_path, map_location="cpu")
    random_test_trajs = load_trajectories(random_test_path, map_location="cpu")

    print(f"Loaded {len(expert_test_trajs)} expert test trajectories from {expert_test_path}")
    print(f"Loaded {len(random_test_trajs)} random test trajectories from {random_test_path}")

    return expert_test_trajs, random_test_trajs


def save_evaluation_report(
    metrics: dict,
    report_dir: str | Path,
) -> Path:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=float)

    return output_path


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Fisher-NHD or ML-IRL checkpoint.",
    )

    parser.add_argument(
        "--env",
        choices=sorted(SUPPORTED_ENVS),
        required=True,
        help="Environment name.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the evaluation config.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/metrics",
        help="Directory where a timestamped evaluation report will be saved.",
    )
    parser.add_argument(
        "--n-agent-traj",
        type=int,
        default=None,
        help="Override the number of agent evaluation trajectories.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only load models and data, without rollouts or metric computation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)

    env_cfg = config["env"]
    eval_cfg = config["evaluation"]
    bootstrap_cfg = eval_cfg.get("bootstrap", {})

    env_name = env_cfg["name"]

    if env_name != args.env:
        raise ValueError(f"Environment mismatch: --env={args.env}, but config contains env.name={env_name}.")

    method, agent = resolve_method_agent(checkpoint)
    env_builders = import_env_builders(env_name)

    print(f"env={env_name} | method={method} | agent={agent}")
    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"report dir: {args.report_dir}")

    random_seed = int(eval_cfg["random_seed"])
    env_seed = int(eval_cfg["env_seed"])

    set_random_seed(random_seed)

    env = env_builders.build_env(
        env_cfg=env_cfg,
        seed=env_seed,
    )

    try:
        policy = env_builders.build_policy(
            env=env,
            policy_cfg=config["policy"],
        )
        reward = env_builders.build_reward(
            env=env,
            reward_cfg=config["reward"],
        )

        policy.load_state_dict(checkpoint["policy_state_dict"])
        reward.load_state_dict(checkpoint["reward_state_dict"])

        policy.eval()
        reward.eval()

        print("Models loaded OK.")

        expert_test_trajs, random_test_trajs = load_test_trajectories(config)

        if args.check_only:
            print("--check-only: static check passed. Skipping rollouts and metric computation.")
            return

        n_agent_traj = args.n_agent_traj or int(eval_cfg["n_agent_traj"])

        agent_test_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_traj,
            deterministic=False,
            desc="Agent rollout",
        )

        policy_nll_value = float(policy_nll(policy, expert_test_trajs))
        rank_corr_value = float(rank_corr(reward, expert_test_trajs + random_test_trajs))

        agent_return = float(env_reward(agent_test_trajs))
        expert_return = float(env_reward(expert_test_trajs))
        random_return = float(env_reward(random_test_trajs))

        bootstrap_metrics = compute_bootstrap_metrics(
            policy=policy,
            reward=reward,
            agent_test_trajs=agent_test_trajs,
            expert_test_trajs=expert_test_trajs,
            random_test_trajs=random_test_trajs,
            bootstrap_cfg=bootstrap_cfg,
        )

        metrics = {
            "env": env_name,
            "method": method,
            "agent": agent,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "evaluation": {
                "random_seed": random_seed,
                "env_seed": env_seed,
                "n_agent_traj": n_agent_traj,
                "bootstrap": bootstrap_cfg,
            },
            "metrics": {
                "PolicyNLL": metric_result(policy_nll_value, bootstrap_metrics, "PolicyNLL"),
                "EnvReward": metric_result(agent_return, bootstrap_metrics, "EnvReward"),
                "ExpertRet": metric_result(expert_return, bootstrap_metrics, "ExpertRet"),
                "RandomRet": metric_result(random_return, bootstrap_metrics, "RandomRet"),
                "RankCorr": metric_result(rank_corr_value, bootstrap_metrics, "RankCorr"),
            },
        }

        print("\n=== Evaluation metrics ===")
        print(format_metric("PolicyNLL", policy_nll_value, bootstrap_metrics))
        print(format_metric("EnvReward", agent_return, bootstrap_metrics))
        print(format_metric("ExpertRet", expert_return, bootstrap_metrics))
        print(format_metric("RandomRet", random_return, bootstrap_metrics))
        print(format_metric("RankCorr", rank_corr_value, bootstrap_metrics))

        report_path = save_evaluation_report(
            metrics=metrics,
            report_dir=args.report_dir,
        )

        print(f"\nSaved metrics: {report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
