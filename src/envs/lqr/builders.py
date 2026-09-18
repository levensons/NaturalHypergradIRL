from src.envs.lqr.env import LQR
from src.envs.lqr.models import Policy, Reward


def build_env(env_cfg: dict, seed: int, custom_reward_fn=None):
    return LQR(
        A=env_cfg["A"],
        B=env_cfg["B"],
        Q=env_cfg["Q"],
        R=env_cfg["R"],
        process_cov=env_cfg["process_cov"],
        initial_cov=env_cfg["initial_cov"],
        max_episode_steps=int(env_cfg["max_steps"]),
        action_limit=float(env_cfg["action_limit"]),
        observation_limit=float(env_cfg["observation_limit"]),
        termination_state_norm=float(env_cfg["termination_state_norm"]),
        reward_scale=float(env_cfg["reward_scale"]),
        seed=seed,
        custom_reward_fn=custom_reward_fn,
    )


def build_policy(env, policy_cfg: dict):
    return Policy(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        hidden_dim=int(policy_cfg["hidden_dim"]),
        n_hidden_layers=int(policy_cfg["n_hidden_layers"]),
        log_std_min=float(policy_cfg["log_std_min"]),
        log_std_max=float(policy_cfg["log_std_max"]),
    )


def build_reward(env, reward_cfg: dict):
    return Reward(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(reward_cfg["hidden_dim"]),
        n_hidden_layers=int(reward_cfg["n_hidden_layers"]),
        clamp_magnitude=float(reward_cfg["clamp_magnitude"]),
    )
