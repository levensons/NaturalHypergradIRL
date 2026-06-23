"""
Запуск:
    python -m src.irl.hopper.ttsa
    python -m src.irl.hopper.ttsa --config config/hopper.yaml
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.agents.sac import SACInnerOptimizer
from src.evaluation.metrics import policy_nll, rank_corr
from src.irl.hopper.fisher import Policy as SACPolicy
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import get_env_dims
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_env_seed, set_random_seed
from src.utils.trajectories import mean_trajectory_length, mean_trajectory_return


Policy = SACPolicy


class LegacyGaussianPolicyNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 64,
        action_scale: float = 1.0,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    # Возвращает среднее и стандартное отклонение старой гауссовой политики.
    def forward(self, states: torch.Tensor):
        mu = self.action_scale * torch.tanh(self.mu_net(states))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mu)
        return mu, std

    # Считает логарифм вероятности действий для старой политики.
    def log_prob(self, states: torch.Tensor, actions: torch.Tensor):
        mu, std = self.forward(states)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        return Normal(mu, std).log_prob(actions).sum(dim=-1)

    # Семплирует действие из старой политики и обрезает его в допустимый диапазон.
    def sample_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mu, std = self.forward(state_t)
            action = Normal(mu, std).sample()
        return np.clip(action.numpy(), -self.action_scale, self.action_scale)


class LegacyRewardNet(nn.Module):
    def __init__(self, state_dim: int, gamma: float = 0.99):
        super().__init__()
        self.net = nn.Linear(state_dim, 1, bias=True)
        self.gamma = gamma

    # Считает недисконтированную обученную награду по состояниям.
    def base_reward(self, states: torch.Tensor, actions=None) -> torch.Tensor:
        return self.net(states).squeeze(-1)

    # Считает обученную награду с дисконтирующим множителем gamma^t.
    def discounted_rewards(self, states: torch.Tensor, actions=None) -> torch.Tensor:
        ts = torch.arange(states.size(0), dtype=torch.float32, device=states.device)
        return self.base_reward(states, actions) * (self.gamma**ts)

    # Считает суммарный дисконтированный возврат старой сети награды.
    def trajectory_return(self, states: torch.Tensor, actions=None) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()

    # Считает награду для переданных состояний через старую сеть награды.
    def forward(self, states: torch.Tensor, actions=None) -> torch.Tensor:
        return self.base_reward(states, actions)


LegacyPolicy = LegacyGaussianPolicyNet
LegacyReward = LegacyRewardNet


class Reward(nn.Module):
    # Создает текущую сеть награды ML-IRL: многослойный перцептрон по паре
    # состояние-действие или только по состоянию.
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 64,
        gamma: float = 0.99,
        state_only: bool = False,
    ):
        super().__init__()

        self.state_only = state_only
        input_dim = state_dim if state_only else state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

        self.init_weights()

    # Инициализирует веса сети награды устойчивыми начальными значениями.
    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        last_layer = self.net[-1]
        nn.init.uniform_(last_layer.weight, -1e-3, 1e-3)
        nn.init.zeros_(last_layer.bias)

    # Считает обученную награду r(s, a) или r(s), если включен режим "только состояния".
    def forward(self, states: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        if self.state_only:
            inputs = states
        else:
            if actions is None:
                raise ValueError("State-action reward requires actions.")
            inputs = torch.cat([states, actions], dim=-1)

        return self.net(inputs).squeeze(-1)

    # Возвращает последовательность обученных наград, умноженных на gamma^t.
    def discounted_rewards(self, states: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        rewards = self.forward(states, actions)
        ts = torch.arange(states.size(0), dtype=torch.float32, device=states.device)
        return torch.pow(self.gamma.to(states.device), ts) * rewards

    # Считает суммарный дисконтированный обученный возврат траектории.
    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        return self.discounted_rewards(states, actions).sum()

    # Считает нормированный обученный возврат, чтобы при необходимости убрать влияние длины.
    def normalized_trajectory_return(self, states: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        discounted = self.discounted_rewards(states, actions)
        ts = torch.arange(states.size(0), dtype=torch.float32, device=states.device)
        denom = torch.pow(self.gamma.to(states.device), ts).sum().clamp_min(1e-8)
        return discounted.sum() / denom


# Собирает траектории текущей политики в Hopper для обновления сети награды или диагностики.
def collect_policy_trajectories(
    env,
    policy: Policy,
    n: int,
    max_steps: int,
    deterministic: bool,
):
    trajs = []

    for _ in range(n):
        states = []
        actions = []
        env_rewards = []

        state, _ = env.reset()

        for _ in range(max_steps):
            action = policy.sample_action(state, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(torch.as_tensor(state, dtype=torch.float32))
            actions.append(torch.as_tensor(action, dtype=torch.float32))
            env_rewards.append(float(reward))

            state = next_state

            if terminated or truncated:
                break

        trajs.append(
            {
                "states": torch.stack(states),
                "actions": torch.stack(actions),
                "env_rewards": torch.tensor(env_rewards, dtype=torch.float32),
            }
        )

    return trajs


# Выбирает n экспертных траекторий для стохастической оценки градиента сети награды.
def sample_trajectories(trajs, n: int):
    if n >= len(trajs):
        return trajs

    idx = np.random.choice(len(trajs), size=n, replace=False)
    return [trajs[int(i)] for i in idx]


# Считает средний обученный возврат по списку траекторий.
def mean_learned_return(reward: Reward, trajs, normalize: bool) -> torch.Tensor:
    # h(theta; tau) из ML-IRL: дисконтированная обученная награда вдоль траектории.
    values = []

    for traj in trajs:
        if normalize:
            values.append(reward.normalized_trajectory_return(traj["states"], traj["actions"]))
        else:
            values.append(reward.trajectory_return(traj["states"], traj["actions"]))

    return torch.stack(values).mean()


# Считает L_outer: средний NLL экспертных действий под текущей политикой.
def normalized_outer_loss(policy: Policy, expert_trajs) -> torch.Tensor:
    losses = []

    for traj in expert_trajs:
        losses.append(-policy.log_prob(traj["states"], traj["actions"]).mean())

    return torch.stack(losses).mean()


# Считает L_inner-прокси для текущей политики и обученной награды на траекториях агента.
def normalized_inner_loss(policy: Policy, reward: Reward, agent_trajs) -> torch.Tensor:
    losses = []

    for traj in agent_trajs:
        states = traj["states"]
        actions = traj["actions"]
        losses.append((policy.log_prob(states, actions) - reward.discounted_rewards(states, actions)).mean())

    return torch.stack(losses).mean()


class RewardOptimizer:
    # Создает оптимизатор для медленного обновления сети награды в TTSA/ML-IRL.
    def __init__(
        self,
        reward: Reward,
        lr: float,
        max_grad_norm: float,
        weight_decay: float,
        normalize_returns: bool,
    ):
        self.reward = reward
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.normalize_returns = normalize_returns
        self.optimizer = torch.optim.Adam(self.reward.parameters(), lr=lr)
        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

    # Делает одно обновление сети награды по разности обученных возвратов эксперта и агента.
    def step(self, expert_trajs, agent_trajs) -> dict:
        # Обновление сети награды из Алгоритма 2:
        # максимизируем h(expert) - h(agent), чтобы экспертные траектории
        # получали больше обученной награды, чем траектории текущей политики.
        expert_value = mean_learned_return(self.reward, expert_trajs, self.normalize_returns)
        agent_value = mean_learned_return(self.reward, agent_trajs, self.normalize_returns)
        l2 = torch.zeros((), dtype=torch.float32)

        for param in self.reward.parameters():
            l2 = l2 + param.pow(2).mean()

        # Adam минимизирует функцию потерь, поэтому этот знак соответствует подъему по
        # h(expert) - h(agent), то есть оценке градиента сети награды из ML-IRL.
        loss = agent_value - expert_value + self.weight_decay * l2

        self.optimizer.zero_grad()
        loss.backward()

        params = [p for p in self.reward.parameters() if p.grad is not None]
        if params:
            self.raw_grad_norm = float(torch.norm(torch.stack([p.grad.detach().norm() for p in params])).item())
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self.clipped_grad_norm = float(torch.norm(torch.stack([p.grad.detach().norm() for p in params])).item())
        else:
            self.raw_grad_norm = 0.0
            self.clipped_grad_norm = 0.0

        self.optimizer.step()

        return {
            "reward_loss": float(loss.item()),
            "expert_learned_return": float(expert_value.detach().item()),
            "agent_learned_return": float(agent_value.detach().item()),
            "reward_grad_raw": self.raw_grad_norm,
            "reward_grad_clipped": self.clipped_grad_norm,
        }


# Создает внутренний SAC-оптимизатор, который быстро улучшает политику под текущую сеть награды.
def make_sac_inner_optimizer(
    sac_env,
    env,
    reward: Reward,
    policy: Policy,
    state_dim: int,
    action_dim: int,
    ttsa_cfg: dict,
):
    sac_cfg = ttsa_cfg["sac"]

    return SACInnerOptimizer(
        env=sac_env,
        reward=reward,
        policy=policy,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        lr_actor=float(sac_cfg["lr_actor"]),
        lr_q=float(sac_cfg["lr_q"]),
        buffer_size=int(sac_cfg["buffer_size"]),
        batch_size=int(sac_cfg["batch_size"]),
        learning_starts=int(sac_cfg["learning_starts"]),
        gamma=float(sac_cfg["gamma"]),
        tau=float(sac_cfg["tau"]),
        alpha=float(sac_cfg["alpha"]),
        autotune=bool(sac_cfg["autotune"]),
        policy_frequency=int(sac_cfg["policy_frequency"]),
        target_network_frequency=int(sac_cfg["target_network_frequency"]),
    )


# Обновляет экспоненциальное скользящее среднее метрики.
def update_ema(previous: float | None, value: float, beta: float) -> float:
    if previous is None:
        return value

    return beta * previous + (1.0 - beta) * value


# Запускает полный TTSA/ML-IRL цикл обучения для Hopper.
def train_ttsa(env, config: dict, logger) -> dict:
    ttsa_cfg = config["ttsa"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    data_cfg = config["data"]
    ckpt_cfg = config["checkpoint"]

    expert_train_path = Path(data_cfg["expert_train_trajs"])
    expert_valid_path = Path(data_cfg["expert_valid_trajs"])
    random_valid_path = Path(data_cfg["random_valid_trajs"])

    expert_train_trajs = load_trajectories(expert_train_path, map_location="cpu")
    expert_valid_trajs = load_trajectories(expert_valid_path, map_location="cpu")
    random_valid_trajs = load_trajectories(random_valid_path, map_location="cpu")

    n_expert_train_trajs = int(ttsa_cfg.get("n_expert_train_trajs", len(expert_train_trajs)))
    if n_expert_train_trajs <= 0:
        raise ValueError("ttsa.n_expert_train_trajs must be positive.")
    expert_train_trajs = expert_train_trajs[:n_expert_train_trajs]

    logger.info(f"Loaded {len(expert_train_trajs)} expert train trajectories from {expert_train_path}")
    logger.info(f"Loaded {len(expert_valid_trajs)} expert valid trajectories from {expert_valid_path}")
    logger.info(f"Loaded {len(random_valid_trajs)} random valid trajectories from {random_valid_path}")

    state_dim, action_dim = get_env_dims(env)
    max_steps = int(config["env"]["max_steps"])

    hidden = int(policy_cfg["hidden"])
    n_layers = int(policy_cfg["n_hidden_layers"])
    gamma = float(reward_cfg["gamma"])

    n_outer_steps = int(ttsa_cfg["n_outer_steps"])
    sac_steps_per_iter = int(ttsa_cfg["sac_steps_per_iter"])
    reward_batch_trajs = int(ttsa_cfg["reward_batch_trajs"])
    eval_trajs = int(ttsa_cfg["eval_trajs"])
    metrics_every = int(ttsa_cfg["metrics_every"])
    deterministic_eval = bool(ttsa_cfg["deterministic_eval"])
    ema_beta = float(ttsa_cfg["ema_beta"])

    reward = Reward(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=int(reward_cfg["hidden"]),
        gamma=gamma,
        state_only=bool(ttsa_cfg["state_only_reward"]),
    )
    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden=hidden,
        n_hidden_layers=n_layers,
        log_std_min=float(policy_cfg["log_std_min"]),
        log_std_max=float(policy_cfg["log_std_max"]),
    )

    reward_optimizer = RewardOptimizer(
        reward=reward,
        lr=float(ttsa_cfg["reward_lr"]),
        max_grad_norm=float(ttsa_cfg["reward_max_grad_norm"]),
        weight_decay=float(ttsa_cfg["reward_weight_decay"]),
        normalize_returns=bool(ttsa_cfg["normalize_reward_returns"]),
    )

    sac_env = gym.make(config["env"]["id"])
    set_env_seed(sac_env, int(ttsa_cfg["sac_env_seed"]))
    inner_optimizer = make_sac_inner_optimizer(
        sac_env=sac_env,
        env=env,
        reward=reward,
        policy=policy,
        state_dim=state_dim,
        action_dim=action_dim,
        ttsa_cfg=ttsa_cfg,
    )

    history = {
        "l_outer": [],
        "l_outer_ema": [],
        "l_inner": [],
        "l_inner_ema": [],
        "agent_return": [],
        "agent_len": [],
        "rank_corr": [],
        "policy_nll": [],
        "reward_loss": [],
        "reward_grad_raw": [],
        "reward_grad_clipped": [],
        "iter_time": [],
    }

    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = str(ckpt_dir / "ttsa_sac.pt")
    best_louter_checkpoint_path = str(ckpt_dir / "ttsa_sac_best_louter.pt")
    best_l_outer_ema = float("inf")
    l_outer_ema = None
    l_inner_ema = None

    arch = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "policy_hidden": hidden,
        "policy_n_hidden_layers": n_layers,
        "reward_hidden": int(reward_cfg["hidden"]),
        "reward_gamma": gamma,
        "reward_state_only": bool(ttsa_cfg["state_only_reward"]),
        "log_std_min": float(policy_cfg["log_std_min"]),
        "log_std_max": float(policy_cfg["log_std_max"]),
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "method": "ttsa",
        "agent": "sac",
        "env_name": config["env"]["name"],
        "env_id": config["env"]["id"],
        "action_type": config["env"]["action_type"],
    }

    header = (
        f"{'Step':>5} | {'L_outer':>9} | {'L_out_ema':>9} | {'L_inner':>9} | "
        f"{'L_in_ema':>9} | {'agent_ret':>9} | {'RankCorr':>8} | "
        f"{'PolicyNLL':>10} | {'R_grad':>8} | {'time':>7}"
    )
    logger.info(header)

    t_start = time.time()

    try:
        for step in range(1, n_outer_steps + 1):
            t_iter = time.time()

            # Быстрая шкала времени: улучшаем политику несколькими SAC-шагами
            # под текущей обученной наградой r(s, a; theta).
            inner_optimizer.optimize(n_steps=sac_steps_per_iter, log_every=0)

            # Семплируем две пачки траекторий для стохастической оценки
            # градиента сети награды ML-IRL: h(expert) - h(agent).
            reward_agent_trajs = collect_policy_trajectories(
                env=env,
                policy=policy,
                n=reward_batch_trajs,
                max_steps=max_steps,
                deterministic=False,
            )
            reward_expert_trajs = sample_trajectories(expert_train_trajs, reward_batch_trajs)

            # Медленная шкала времени: одно обновление сети награды.
            reward_stats = reward_optimizer.step(reward_expert_trajs, reward_agent_trajs)

            if step % metrics_every == 0 or step == 1:
                # Эти траектории нужны только для диагностики и чекпойнтов;
                # они не используются в обновлении сети награды выше.
                eval_agent_trajs = collect_policy_trajectories(
                    env=env,
                    policy=policy,
                    n=eval_trajs,
                    max_steps=max_steps,
                    deterministic=deterministic_eval,
                )

                l_outer = float(normalized_outer_loss(policy, expert_valid_trajs).item())
                l_inner = float(normalized_inner_loss(policy, reward, eval_agent_trajs).item())
                l_outer_ema = update_ema(l_outer_ema, l_outer, ema_beta)
                l_inner_ema = update_ema(l_inner_ema, l_inner, ema_beta)

                agent_ret = mean_trajectory_return(eval_agent_trajs)
                agent_len = mean_trajectory_length(eval_agent_trajs)
                rank_corr_val = rank_corr(reward, expert_valid_trajs + random_valid_trajs)
                policy_nll_val = policy_nll(policy, expert_valid_trajs)
                iter_elapsed = time.time() - t_iter

                history["l_outer"].append(l_outer)
                history["l_outer_ema"].append(l_outer_ema)
                history["l_inner"].append(l_inner)
                history["l_inner_ema"].append(l_inner_ema)
                history["agent_return"].append(agent_ret)
                history["agent_len"].append(agent_len)
                history["rank_corr"].append(rank_corr_val)
                history["policy_nll"].append(policy_nll_val)
                history["reward_loss"].append(reward_stats["reward_loss"])
                history["reward_grad_raw"].append(reward_stats["reward_grad_raw"])
                history["reward_grad_clipped"].append(reward_stats["reward_grad_clipped"])
                history["iter_time"].append(iter_elapsed)

                if l_outer_ema < best_l_outer_ema:
                    # Диагностический чекпойнт: лучшая валидационная ошибка имитации.
                    best_l_outer_ema = l_outer_ema
                    save_checkpoint(
                        path=best_louter_checkpoint_path,
                        policy=policy,
                        reward=reward,
                        arch=arch,
                        step=step,
                        best_l_outer_ema=best_l_outer_ema,
                        agent_return=agent_ret,
                    )

                save_checkpoint(
                    # Основной чекпойнт: текущая/последняя пара политика-награда
                    # для базового метода как в статье, без выбора лучшего возврата.
                    path=best_checkpoint_path,
                    policy=policy,
                    reward=reward,
                    arch=arch,
                    step=step,
                    l_outer_ema=l_outer_ema,
                    l_inner_ema=l_inner_ema,
                    agent_return=agent_ret,
                )

                row = (
                    f"{step:>5} | {l_outer:>9.4f} | {l_outer_ema:>9.4f} | "
                    f"{l_inner:>9.4f} | {l_inner_ema:>9.4f} | {agent_ret:>9.1f} | "
                    f"{rank_corr_val:>8.3f} | {policy_nll_val:>10.2f} | "
                    f"{reward_stats['reward_grad_clipped']:>8.3f} | {iter_elapsed:>6.1f}s"
                )
                logger.info(row)

    finally:
        sac_env.close()

    total_time = time.time() - t_start
    logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    return history


# Разбирает аргументы командной строки для запуска TTSA.
def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Стабильный TTSA / ML-IRL SAC для Hopper")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


# Точка входа: загружает конфиг, создает среду, запускает обучение и сохраняет историю.
def main() -> None:
    args = parse()

    config_path = resolve_config_path("hopper", args.config)
    config = load_config(config_path)

    ttsa_cfg = config["ttsa"]
    log_cfg = config["logging"]

    logger = get_logger("ttsa_hopper", log_dir=log_cfg["log_dir"])

    set_random_seed(int(ttsa_cfg["random_seed"]))
    env = gym.make(config["env"]["id"])
    set_env_seed(env, int(ttsa_cfg["env_seed"]))

    try:
        logger.info("=== Stable TTSA / ML-IRL Hopper SAC ===")
        history = train_ttsa(env, config, logger)
        report_path = Path(log_cfg["report_dir"]) / "ttsa_hopper_history.json"
        save_history(history, str(report_path))
        logger.info(f"History saved to {report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
