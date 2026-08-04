"""
Evaluate a saved IRL checkpoint.

Usage:
    python -m src.evaluation.evaluate --env cartpole --checkpoint checkpoints/cartpole/fisher.pt
    python -m src.evaluation.evaluate --env hopper   --checkpoint checkpoints/hopper/fisher.pt

Static check only, without rollouts or metric computation:
    python -m src.evaluation.evaluate --env cartpole --checkpoint checkpoints/cartpole/fisher.pt --check-only
"""

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np

from src.evaluation.bootstrap import bootstrap_metric, bootstrap_two_group_metric
from src.evaluation.metrics import env_reward, policy_nll, rank_corr
from src.irl.lqr.env import LQR
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import Environment
from src.utils.seeding import set_random_seed
from src.utils.trajectories import collect_trajectories


SUPPORTED_ENVS = {"cartpole", "hopper", "lqr"}
SUPPORTED_METHODS = {"fisher", "ml_irl"}
SUPPORTED_AGENTS = {"reinforce", "sac"}


def build_env(env_cfg: dict, seed: int):
    if env_cfg["name"] == "lqr":
        return LQR(
            A=env_cfg["A"],
            B=env_cfg["B"],
            Q=env_cfg["Q"],
            R=env_cfg["R"],
            process_cov=env_cfg["process_cov"],
            initial_cov=env_cfg["initial_cov"],
            seed=seed,
            max_episode_steps=int(env_cfg["max_steps"]),
            action_limit=float(env_cfg["action_limit"]),
            observation_limit=float(env_cfg["observation_limit"]),
            termination_state_norm=float(env_cfg["termination_state_norm"]),
            reward_scale=float(env_cfg["reward_scale"]),
            custom_reward_fn=None,
        )

    return Environment(
        id=env_cfg["id"],
        seed=seed,
        max_episode_steps=int(env_cfg["max_steps"]),
    )


def import_models_module(env_name: str) -> tuple[ModuleType, str]:
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported environment: {env_name}. Available: {sorted(SUPPORTED_ENVS)}")

    module_path = f"src.irl.{env_name}.models"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        if error.name == module_path:
            raise ModuleNotFoundError(f"Models module does not exist: {module_path}") from error

        # A dependency imported inside models.py is missing.
        raise

    if not hasattr(module, "Policy"):
        raise AttributeError(f"{module_path} does not define `Policy`.")

    if not hasattr(module, "Reward"):
        raise AttributeError(f"{module_path} does not define `Reward`.")

    return module, module_path


def infer_method_agent(checkpoint_path: str | Path) -> tuple[str, str]:
    stem = Path(checkpoint_path).stem.lower()

    if stem.startswith("ml_irl"):
        agent = "sac" if "sac" in stem else "reinforce"
        return "ml_irl", agent

    if stem.startswith("fisher"):
        agent = "sac" if "sac" in stem else "reinforce"
        return "fisher", agent

    return "fisher", "reinforce"


def resolve_method_agent(
    args: argparse.Namespace,
    arch: dict,
) -> tuple[str, str]:
    inferred_method, inferred_agent = infer_method_agent(args.checkpoint)

    method = args.method or arch.get("method") or inferred_method
    agent = args.agent or arch.get("agent") or inferred_agent

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}. Available: {sorted(SUPPORTED_METHODS)}")

    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported agent: {agent}. Available: {sorted(SUPPORTED_AGENTS)}")

    return method, agent


def resolve_action_bounds(
    arch: dict,
    env,
) -> tuple[np.ndarray, np.ndarray]:
    if "action_low" in arch:
        action_low = np.asarray(arch["action_low"], dtype=np.float32)
    elif hasattr(env, "action_low"):
        action_low = np.asarray(env.action_low, dtype=np.float32)
    elif hasattr(env, "action_space") and hasattr(env.action_space, "low"):
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
    else:
        raise ValueError("Could not determine the lower action bound from the checkpoint or environment.")

    if "action_high" in arch:
        action_high = np.asarray(arch["action_high"], dtype=np.float32)
    elif hasattr(env, "action_high"):
        action_high = np.asarray(env.action_high, dtype=np.float32)
    elif hasattr(env, "action_space") and hasattr(env.action_space, "high"):
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
    else:
        raise ValueError("Could not determine the upper action bound from the checkpoint or environment.")

    return action_low, action_high


def build_policy(
    models_module: ModuleType,
    arch: dict,
    env,
    env_cfg: dict,
):
    state_dim = int(arch["state_dim"])
    action_dim = int(arch["action_dim"])
    hidden_dim = int(arch.get("policy_hidden", arch.get("policy_hidden_dim", 64)))
    n_hidden_layers = int(arch.get("policy_n_hidden_layers", 2))

    action_type = env_cfg["action_type"]

    if action_type == "discrete":
        return models_module.Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    if action_type == "continuous":
        action_low, action_high = resolve_action_bounds(
            arch=arch,
            env=env,
        )

        return models_module.Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            log_std_min=float(arch.get("log_std_min", -20.0)),
            log_std_max=float(arch.get("log_std_max", 2.0)),
        )

    raise ValueError(f"Unsupported action type: {action_type}. Expected `discrete` or `continuous`.")


def build_reward(
    models_module: ModuleType,
    arch: dict,
):
    kwargs = {
        "state_dim": int(arch["state_dim"]),
        "action_dim": int(arch["action_dim"]),
        "hidden_dim": int(arch.get("reward_hidden", arch.get("reward_hidden_dim", 64))),
        "n_hidden_layers": int(arch.get("reward_n_hidden_layers", 2)),
    }

    if "reward_clamp_magnitude" in arch:
        kwargs["clamp_magnitude"] = float(arch["reward_clamp_magnitude"])

    return models_module.Reward(**kwargs)


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
    result = {
        "value": float(value),
    }

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

    expert_test_trajs = load_trajectories(
        expert_test_path,
        map_location="cpu",
    )
    random_test_trajs = load_trajectories(
        random_test_path,
        map_location="cpu",
    )

    print(f"Loaded {len(expert_test_trajs)} expert test trajectories from {expert_test_path}")
    print(f"Loaded {len(random_test_trajs)} random test trajectories from {random_test_path}")

    return expert_test_trajs, random_test_trajs


def save_evaluation_report(
    metrics: dict,
    config: dict,
    env_name: str,
    method: str,
) -> Path:
    report_root = Path(config["logging"]["report_dir"])
    report_dir = report_root / "metrics" / env_name / method
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"{timestamp}.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=float)

    return report_path


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Fisher-NHD or ML-IRL checkpoint.",
    )

    parser.add_argument(
        "--env",
        choices=sorted(SUPPORTED_ENVS),
        default=None,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Explicit path to config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the .pt checkpoint file.",
    )
    parser.add_argument(
        "--method",
        choices=sorted(SUPPORTED_METHODS),
        default=None,
    )
    parser.add_argument(
        "--agent",
        choices=sorted(SUPPORTED_AGENTS),
        default=None,
    )
    parser.add_argument(
        "--n-agent-traj",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only load models and data, without rollouts or metric computation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse()

    config_path = resolve_config_path(
        args.env,
        args.config,
    )
    config = load_config(config_path)

    env_cfg = config["env"]
    eval_cfg = config["evaluation"]
    bootstrap_cfg = eval_cfg.get("bootstrap", {})

    checkpoint = load_checkpoint(args.checkpoint)
    arch = checkpoint["arch"]

    env_name = env_cfg["name"]
    method, agent = resolve_method_agent(
        args=args,
        arch=arch,
    )

    models_module, module_path = import_models_module(env_name)

    print(f"env={env_name} | method={method} | agent={agent}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"models module: {module_path}")

    random_seed = int(eval_cfg["random_seed"])
    env_seed = int(eval_cfg["env_seed"])

    set_random_seed(random_seed)

    env = build_env(
        env_cfg=env_cfg,
        seed=env_seed,
    )

    try:
        policy = build_policy(
            models_module=models_module,
            arch=arch,
            env=env,
            env_cfg=env_cfg,
        )
        reward = build_reward(
            models_module=models_module,
            arch=arch,
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

        policy_nll_value = policy_nll(
            policy,
            expert_test_trajs,
        )
        rank_corr_value = rank_corr(
            reward,
            expert_test_trajs + random_test_trajs,
        )

        agent_return = env_reward(agent_test_trajs)
        expert_return = env_reward(expert_test_trajs)
        random_return = env_reward(random_test_trajs)

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
            "checkpoint": str(args.checkpoint),
            "config": str(config_path),
            "models_module": module_path,
            "evaluation": {
                "random_seed": random_seed,
                "env_seed": env_seed,
                "n_agent_traj": n_agent_traj,
                "bootstrap": bootstrap_cfg,
            },
            "metrics": {
                "PolicyNLL": metric_result(
                    policy_nll_value,
                    bootstrap_metrics,
                    "PolicyNLL",
                ),
                "EnvReward": metric_result(
                    agent_return,
                    bootstrap_metrics,
                    "EnvReward",
                ),
                "ExpertRet": metric_result(
                    expert_return,
                    bootstrap_metrics,
                    "ExpertRet",
                ),
                "RandomRet": metric_result(
                    random_return,
                    bootstrap_metrics,
                    "RandomRet",
                ),
                "RankCorr": metric_result(
                    rank_corr_value,
                    bootstrap_metrics,
                    "RankCorr",
                ),
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
            config=config,
            env_name=env_name,
            method=method,
        )

        print(f"\nSaved report: {report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()