"""
Evaluate a saved IRL checkpoint.

Usage:
    python -m src.evaluation.evaluate --env cartpole --checkpoint checkpoints/cartpole/fisher_reinforce.pt
    python -m src.evaluation.evaluate --env hopper   --checkpoint checkpoints/hopper/fisher_sac.pt

Static check only, without rollouts or metric computation:
    python -m src.evaluation.evaluate --env cartpole --checkpoint checkpoints/cartpole/fisher_reinforce.pt --check-only
"""

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.evaluation.metrics import env_reward, policy_nll, rank_corr
from src.evaluation.bootstrap import bootstrap_metric, bootstrap_two_group_metric
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_expert_test_trajectories, load_random_test_trajectories
from src.utils.seeding import set_random_seed, set_env_seed
from src.utils.trajectories import collect_trajectories

_REGISTRY: dict[tuple[str, str, str], str] = {
    ("cartpole", "fisher", "reinforce"): "src.irl.cartpole.fisher_cartpole",
    ("cartpole", "ttsa", "reinforce"): "src.irl.cartpole.ttsa_cartpole",
    ("hopper", "fisher", "reinforce"): "src.irl.hopper.fisher_reinforce_hopper",
    ("hopper", "fisher", "sac"): "src.irl.hopper.fisher_sac_hopper",
    ("hopper", "ttsa", "reinforce"): "src.irl.hopper.ttsa_hopper",
}


def infer_method_agent(checkpoint_path: str | Path) -> tuple[str, str]:
    stem = Path(checkpoint_path).stem
    parts = stem.split("_")

    method = parts[0] if parts else "fisher"
    agent = parts[1] if len(parts) > 1 else "reinforce"

    return method, agent


def build_policy(module, arch: dict, env, env_cfg: dict, method: str, agent: str):
    state_dim = arch["state_dim"]
    action_dim = arch["action_dim"]
    hidden = arch.get("policy_hidden", 64)
    n_layers = arch.get("policy_n_hidden_layers", 2)

    if method == "ttsa" and env_cfg["name"] == "hopper":
        return module.Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=hidden,
            action_scale=1.0,
        )

    if method == "ttsa" and env_cfg["name"] == "cartpole":
        return module.Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=hidden,
        )

    if env_cfg.get("action_type") == "continuous" and agent == "sac":
        action_low = np.array(arch["action_low"], dtype=np.float32) if "action_low" in arch else env.action_space.low
        action_high = (
            np.array(arch["action_high"], dtype=np.float32) if "action_high" in arch else env.action_space.high
        )

        return module.Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden=hidden,
            n_hidden_layers=n_layers,
            log_std_min=arch.get("log_std_min", -5),
            log_std_max=arch.get("log_std_max", 2),
        )

    return module.Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=hidden,
        n_hidden_layers=n_layers,
    )


def build_reward(module, arch: dict, method: str):
    if method == "ttsa":
        return module.Reward(
            state_dim=arch["state_dim"],
            gamma=arch.get("reward_gamma", 0.99),
        )

    return module.Reward(
        state_dim=arch["state_dim"],
        action_dim=arch["action_dim"],
        hidden=arch.get("reward_hidden", 64),
        gamma=arch.get("reward_gamma", 0.99),
    )


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

    n_samples = int(bootstrap_cfg.get("n_samples", 1000))
    seed = int(bootstrap_cfg.get("seed", 42))

    return {
        "PolicyNLL": bootstrap_metric(
            metric_fn=lambda trajs: policy_nll(policy, trajs),
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


def format_metric(
    name: str,
    value: float,
    bootstrap_metrics: dict[str, dict[str, float]] | None,
    digits: int = 4,
) -> str:
    if bootstrap_metrics is None or name not in bootstrap_metrics:
        return f"{name:<10} = {value:.{digits}f}"

    stats = bootstrap_metrics[name]

    return f"{name:<10} = {value:.{digits}f} " f"± {stats['std']:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved IRL checkpoint.")

    parser.add_argument("--env", choices=["cartpole", "hopper"], default=None)
    parser.add_argument("--config", default=None, help="Explicit path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--method", choices=["fisher", "ttsa"], default=None)
    parser.add_argument("--agent", choices=["reinforce", "sac"], default=None)
    parser.add_argument("--n-agent-traj", type=int, default=None)

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only load models and data, without rollouts or metric computation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)

    env_cfg = config["env"]
    eval_cfg = config["evaluation"]
    bootstrap_cfg = eval_cfg.get("bootstrap", {})

    method_inferred, agent_inferred = infer_method_agent(args.checkpoint)
    method = args.method or method_inferred
    agent = args.agent or agent_inferred
    env_name = env_cfg["name"]

    key = (env_name, method, agent)
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown combination (env={env_name}, method={method}, agent={agent}). " f"Available: {list(_REGISTRY)}"
        )

    module_path = _REGISTRY[key]

    print(f"env={env_name} | method={method} | agent={agent}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"module: {module_path}")

    random_seed = int(eval_cfg["random_seed"])
    env_seed = int(eval_cfg["env_seed"])

    set_random_seed(random_seed)

    env = gym.make(env_cfg["id"])
    set_env_seed(env, env_seed)

    try:
        ckpt = load_checkpoint(args.checkpoint)
        arch = ckpt["arch"]

        module = importlib.import_module(module_path)

        policy = build_policy(
            module=module,
            arch=arch,
            env=env,
            env_cfg=env_cfg,
            method=method,
            agent=agent,
        )
        reward = build_reward(
            module=module,
            arch=arch,
            method=method,
        )

        policy.load_state_dict(ckpt["policy_state_dict"])
        reward.load_state_dict(ckpt["reward_state_dict"])
        policy.eval()
        reward.eval()

        print("Models loaded OK.")

        expert_test_trajs, expert_test_path = load_expert_test_trajectories(config)
        random_test_trajs, random_test_path = load_random_test_trajectories(config)

        print(f"Loaded {len(expert_test_trajs)} expert test trajs from {expert_test_path}")
        print(f"Loaded {len(random_test_trajs)} random test trajs from {random_test_path}")

        if args.check_only:
            print("--check-only: static check passed. Skipping rollouts and metric computation.")
            return

        max_steps = int(env_cfg["max_steps"])
        n_agent_traj = args.n_agent_traj or int(eval_cfg["n_agent_traj"])

        agent_test_trajs = collect_trajectories(
            env=env,
            policy=policy,
            n=n_agent_traj,
            max_steps=max_steps,
            desc="agent rollout",
        )

        policy_nll_val = policy_nll(policy, expert_test_trajs)
        rank_corr_val = rank_corr(reward, expert_test_trajs + random_test_trajs)

        agent_ret = env_reward(agent_test_trajs)
        expert_ret = env_reward(expert_test_trajs)
        random_ret = env_reward(random_test_trajs)

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
            "evaluation": {
                "random_seed": random_seed,
                "env_seed": env_seed,
                "n_agent_traj": n_agent_traj,
                "bootstrap": bootstrap_cfg,
            },
            "PolicyNLL": policy_nll_val,
            "EnvReward": agent_ret,
            "ExpertRet": expert_ret,
            "RandomRet": random_ret,
            "RankCorr": rank_corr_val,
            "bootstrap": bootstrap_metrics,
        }

        print("\n=== Table II Metrics ===")
        print(format_metric("PolicyNLL", policy_nll_val, bootstrap_metrics))
        print(format_metric("EnvReward", agent_ret, bootstrap_metrics))
        print(format_metric("ExpertRet", expert_ret, bootstrap_metrics))
        print(format_metric("RandomRet", random_ret, bootstrap_metrics))
        print(format_metric("RankCorr", rank_corr_val, bootstrap_metrics))

        report_dir = Path(config.get("logging", {}).get("report_dir", "reports"))
        report_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"{env_name}_{method}_{agent}_{ts}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=float)

        print(f"\nSaved report: {report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
