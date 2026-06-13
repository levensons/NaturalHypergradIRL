"""
CLI для оценки сохранённых чекпоинтов.

Запуск:
  python -m evaluation.evaluate --env cartpole --checkpoint checkpoints/cartpole/fisher_reinforce.pt
  python -m evaluation.evaluate --env hopper   --checkpoint checkpoints/hopper/fisher_sac.pt
  python -m evaluation.evaluate --env cartpole --checkpoint checkpoints/cartpole/fisher_reinforce.pt --check-only

Флаг --check-only: только загрузка моделей и данных без роллаутов (статическая проверка).
"""
import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from evaluation.metrics import PolicyNLL, RankCorr, env_reward
from src.common.checkpoint import load_checkpoint
from src.common.config import load_config, resolve_config_path

# Диспетчер: (env, method, agent) -> путь к модулю с классами Policy и Reward
_REGISTRY: dict[tuple[str, str, str], str] = {
    ("cartpole", "fisher", "reinforce"): "irl.cartpole.fisher_cartpole",
    ("cartpole", "ttsa",   "reinforce"): "irl.cartpole.ttsa_cartpole",
    ("hopper",   "fisher", "reinforce"): "irl.hopper.fisher_reinforce_hopper",
    ("hopper",   "fisher", "sac"):       "irl.hopper.fisher_sac_hopper",
    ("hopper",   "ttsa",   "reinforce"): "irl.hopper.ttsa_hopper",
}


def _infer_method_agent(checkpoint_path: str) -> tuple[str, str]:
    """Парсит method и agent из имени файла, напр. 'fisher_reinforce.pt'."""
    stem = Path(checkpoint_path).stem
    parts = stem.split("_")
    method = parts[0] if parts else "fisher"
    agent = parts[1] if len(parts) > 1 else "reinforce"
    return method, agent


def _build_policy(module, arch: dict, env, env_cfg: dict, agent: str = "reinforce"):
    state_dim = arch["state_dim"]
    action_dim = arch["action_dim"]
    hidden = arch.get("policy_hidden", 64)
    n_layers = arch.get("policy_n_hidden_layers", 2)

    if env_cfg.get("action_type") == "continuous" and agent == "sac":
        log_std_min = arch.get("log_std_min", -5)
        log_std_max = arch.get("log_std_max", 2)
        action_low = np.array(arch["action_low"]) if "action_low" in arch else env.action_space.low
        action_high = np.array(arch["action_high"]) if "action_high" in arch else env.action_space.high
        return module.Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden=hidden,
            n_hidden_layers=n_layers,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
    return module.Policy(state_dim=state_dim, action_dim=action_dim, hidden=hidden, n_hidden_layers=n_layers)


def _build_reward(module, arch: dict):
    return module.Reward(
        state_dim=arch["state_dim"],
        action_dim=arch["action_dim"],
        hidden=arch.get("reward_hidden", 64),
        gamma=arch.get("reward_gamma", 0.99),
    )


def _collect_agent_trajs(env, policy, n: int, max_steps: int):
    trajs = []
    for _ in tqdm(range(n), desc="agent rollout", leave=False):
        states, actions, rewards = [], [], []
        s, _ = env.reset()
        for _ in range(max_steps):
            a = policy.sample_action(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            states.append(torch.tensor(s, dtype=torch.float32))
            if np.isscalar(a) or np.asarray(a).shape == ():
                actions.append(torch.tensor(int(a), dtype=torch.long))
            else:
                actions.append(torch.tensor(np.asarray(a, dtype=np.float32)))
            rewards.append(float(r))
            s = s_next
            if terminated or truncated:
                break
        trajs.append({"states": torch.stack(states), "actions": torch.stack(actions), "env_rewards": rewards})
    return trajs


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved IRL checkpoint.")
    p.add_argument("--env", choices=["cartpole", "hopper"], default=None)
    p.add_argument("--config", default=None, help="Explicit path to config YAML.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--method", choices=["fisher", "ttsa"], default=None)
    p.add_argument("--agent", choices=["reinforce", "sac"], default=None)
    p.add_argument("--n-eval-traj", type=int, default=None)
    p.add_argument("--check-only", action="store_true",
                   help="Static check only: build models and load data without rollouts.")
    return p.parse_args()


def main():
    args = parse()

    config_path = resolve_config_path(args.env, args.config)
    config = load_config(config_path)
    env_cfg = config["env"]

    # Инференс method/agent из имени файла (если не задан явно)
    method_inferred, agent_inferred = _infer_method_agent(args.checkpoint)
    method = args.method or method_inferred
    agent = args.agent or agent_inferred
    env_name = env_cfg["name"]

    key = (env_name, method, agent)
    if key not in _REGISTRY:
        raise ValueError(f"Unknown combination (env={env_name}, method={method}, agent={agent}). "
                         f"Available: {list(_REGISTRY)}")
    module_path = _REGISTRY[key]

    print(f"env={env_name} | method={method} | agent={agent}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"module: {module_path}")

    ckpt = load_checkpoint(args.checkpoint)
    arch = ckpt["arch"]

    # Создаём среду (нужна для action bounds и роллаутов)
    env = gym.make(env_cfg["id"])

    # Импортируем модуль и строим модели
    module = importlib.import_module(module_path)
    policy = _build_policy(module, arch, env, env_cfg, agent=agent)
    reward = _build_reward(module, arch)

    policy.load_state_dict(ckpt["policy_state_dict"])
    reward.load_state_dict(ckpt["reward_state_dict"])
    policy.eval()
    reward.eval()
    print("Models loaded OK.")

    # Загружаем тестовые траектории из data/
    data_cfg = config.get("data", {})
    expert_test_path = data_cfg.get("expert_test_trajs")
    random_test_path = data_cfg.get("random_test_trajs")

    if not expert_test_path or not Path(expert_test_path).exists():
        raise FileNotFoundError(f"expert_test_trajs not found: {expert_test_path}. "
                                "Check config.data.expert_test_trajs.")
    if not random_test_path or not Path(random_test_path).exists():
        raise FileNotFoundError(f"random_test_trajs not found: {random_test_path}. "
                                "Check config.data.random_test_trajs.")

    expert_test_trajs = torch.load(expert_test_path, map_location="cpu", weights_only=False)
    random_test_trajs = torch.load(random_test_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(expert_test_trajs)} expert test + {len(random_test_trajs)} random test trajs.")

    n_eval_traj = args.n_eval_traj or config.get("training", {}).get("n_eval_traj", 100)

    if args.check_only:
        print("--check-only: static check passed. Skipping rollouts and metric computation.")
        env.close()
        return

    # Роллауты агента
    max_steps = env_cfg.get("max_steps", 1000)
    agent_test_trajs = _collect_agent_trajs(env, policy, n_eval_traj, max_steps)

    # Метрики
    expert_slice = expert_test_trajs[:n_eval_traj]
    random_slice = random_test_trajs[:n_eval_traj]
    rank_pool = expert_slice + agent_test_trajs + random_slice

    policy_nll_val = PolicyNLL()(policy, expert_slice)
    rank_corr_val  = RankCorr()(reward, rank_pool)
    agent_ret      = env_reward(agent_test_trajs)
    expert_ret     = env_reward(expert_slice)
    random_ret     = env_reward(random_slice)

    metrics = {
        "env":       env_name,
        "method":    method,
        "agent":     agent,
        "checkpoint": str(args.checkpoint),
        "n_eval_traj": n_eval_traj,
        "PolicyNLL":  policy_nll_val,
        "EnvReward":  agent_ret,
        "ExpertRet":  expert_ret,
        "RandomRet":  random_ret,
        "RankCorr":   rank_corr_val,
    }

    print("\n=== Table II Metrics ===")
    print(f"PolicyNLL  = {policy_nll_val:.4f}")
    print(f"EnvReward  = {agent_ret:.1f}  (expert={expert_ret:.1f}, random={random_ret:.1f})")
    print(f"RankCorr   = {rank_corr_val:.4f}")

    report_dir = Path(config.get("logging", {}).get("report_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"{env_name}_{method}_{agent}_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nSaved report: {report_path}")

    env.close()


if __name__ == "__main__":
    main()
