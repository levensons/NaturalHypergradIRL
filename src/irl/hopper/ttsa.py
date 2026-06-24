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
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, resolve_config_path
from src.utils.data import load_trajectories
from src.utils.env import get_env_dims
from src.utils.logging import get_logger, save_history
from src.utils.seeding import set_env_seed, set_random_seed
from src.utils.trajectories import mean_trajectory_length, mean_trajectory_return


class Policy(nn.Module):
    # Повторяет SAC actor из официального ML-IRL/SpinningUp:
    # MLP 256x256, default init PyTorch и прямой clamp log_std.
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        hidden: int = 256,
        n_hidden_layers: int = 2,
        log_std_max=2,
        log_std_min=-20,
    ):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    # Возвращает mean и log_std гауссовой политики до tanh-преобразования.
    def forward(self, states: torch.Tensor):
        h = self.net(states)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.log_std_min, self.log_std_max)
        return mean, log_std

    # Семплирует действие через reparameterization trick и считает log_prob с tanh-correction.
    def get_action(self, states: torch.Tensor):
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        log_prob = normal.log_prob(pre_tanh).sum(dim=-1)
        correction = 2.0 * (np.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
        log_prob = log_prob - correction.sum(dim=-1)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    # Обратное tanh-преобразование для подсчета log_prob заданного действия.
    @staticmethod
    def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    # Переводит действие из диапазона среды в [-1, 1].
    def action_to_normalized(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_bias) / self.action_scale

    # Считает log_prob действия под текущей squashed Gaussian политикой.
    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        mean, log_std = self.forward(states)
        std = log_std.exp()
        normalized_action = self.action_to_normalized(actions)
        pre_tanh = self.atanh(normalized_action)
        normal = torch.distributions.Normal(mean, std)

        log_prob = normal.log_prob(pre_tanh).sum(dim=-1)
        correction = 2.0 * (np.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
        return log_prob - correction.sum(dim=-1)

    # Возвращает numpy-действие для среды.
    def sample_action(self, state, deterministic: bool = False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(state_tensor)
                action = torch.tanh(mean) * self.action_scale + self.action_bias
            else:
                action, _, _ = self.get_action(state_tensor)

        return action.squeeze(0).cpu().numpy()


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
        clamp_magnitude: float = 10.0,
        hidden_sizes: tuple[int, ...] | None = None,
    ):
        super().__init__()

        self.state_only = state_only
        input_dim = state_dim if state_only else state_dim + action_dim
        hidden_sizes = hidden_sizes or (hidden, hidden)
        self.clamp_magnitude = clamp_magnitude

        # Архитектура повторяет MLPReward из официального ML-IRL:
        # первый линейный слой, затем блоки линейный слой + ReLU
        # и ограничение выхода в диапазоне [-10, 10].
        self.first_fc = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.blocks_list.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.ReLU(),
                )
            )
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    # Считает обученную награду r(s, a) или r(s), если включен режим "только состояния".
    def forward(self, states: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        if self.state_only:
            inputs = states
        else:
            if actions is None:
                raise ValueError("State-action reward requires actions.")
            inputs = torch.cat([states, actions], dim=-1)

        x = self.first_fc(inputs)
        for block in self.blocks_list:
            x = block(x)
        rewards = self.last_fc(x)
        rewards = torch.clamp(rewards, min=-self.clamp_magnitude, max=self.clamp_magnitude)
        return rewards.squeeze(-1)

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
    fixed_horizon: bool = False,
    shift_states: bool = False,
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

            stored_state = next_state if shift_states else state
            states.append(torch.as_tensor(stored_state, dtype=torch.float32))
            actions.append(torch.as_tensor(action, dtype=torch.float32))
            env_rewards.append(float(reward))

            state = next_state

            if (terminated or truncated) and not fixed_horizon:
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


# Считает среднее значение сети награды по всем точкам траекторий,
# как в официальном ML-IRL.
def mean_point_reward(reward: Reward, trajs) -> torch.Tensor:
    values = []

    for traj in trajs:
        values.append(reward(traj["states"], traj["actions"]).reshape(-1))

    return torch.cat(values).mean()


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
        momentum: float,
        horizon: int,
        loss_type: str,
    ):
        self.reward = reward
        self.max_grad_norm = max_grad_norm
        self.normalize_returns = normalize_returns
        self.horizon = horizon
        self.loss_type = loss_type
        self.optimizer = torch.optim.Adam(
            self.reward.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(momentum, 0.999),
        )
        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

    # Делает одно обновление сети награды по разности обученных возвратов эксперта и агента.
    def step(self, expert_trajs, agent_trajs) -> dict:
        if self.loss_type == "official_ml_irl":
            # Официальный ML-IRL использует T * (E_agent[r] - E_expert[r]).
            # Минимизация этого выражения поднимает награду эксперта относительно агента.
            expert_value = mean_point_reward(self.reward, expert_trajs)
            agent_value = mean_point_reward(self.reward, agent_trajs)
            loss = self.horizon * (agent_value - expert_value)
        else:
            # Старый режим оставлен как запасной вариант для экспериментов с дисконтированным h(tau).
            expert_value = mean_learned_return(self.reward, expert_trajs, self.normalize_returns)
            agent_value = mean_learned_return(self.reward, agent_trajs, self.normalize_returns)
            loss = agent_value - expert_value

        self.optimizer.zero_grad()
        loss.backward()

        params = [p for p in self.reward.parameters() if p.grad is not None]
        if params:
            self.raw_grad_norm = float(torch.norm(torch.stack([p.grad.detach().norm() for p in params])).item())
            if self.max_grad_norm > 0.0:
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
        q_hidden=int(sac_cfg.get("q_hidden", 64)),
        default_init=bool(sac_cfg.get("default_init", False)),
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
    agent_train_trajs = int(ttsa_cfg.get("agent_train_trajs", ttsa_cfg["reward_batch_trajs"]))
    expert_resample_trajs = int(ttsa_cfg.get("expert_resample_trajs", ttsa_cfg["reward_batch_trajs"]))
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
        clamp_magnitude=float(ttsa_cfg["reward_clamp_magnitude"]),
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
        momentum=float(ttsa_cfg["reward_momentum"]),
        horizon=max_steps,
        loss_type=str(ttsa_cfg["reward_loss"]),
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
        "reward_clamp_magnitude": float(ttsa_cfg["reward_clamp_magnitude"]),
        "log_std_min": float(policy_cfg["log_std_min"]),
        "log_std_max": float(policy_cfg["log_std_max"]),
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "method": "ttsa",
        "agent": "sac",
        "source_algorithm": "official_ml_irl_maxentirl_sa",
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
                n=agent_train_trajs,
                max_steps=max_steps,
                deterministic=False,
                fixed_horizon=True,
                shift_states=True,
            )
            reward_expert_trajs = sample_trajectories(expert_train_trajs, expert_resample_trajs)

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
