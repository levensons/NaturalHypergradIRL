from src.envs.cartpole.models import Policy, Reward
from src.utils.env import Environment


def build_env(env_cfg: dict, seed: int, custom_reward_fn=None):
    return Environment(
        id=env_cfg["id"],
        seed=seed,
        max_episode_steps=int(env_cfg["max_steps"]),
        custom_reward_fn=custom_reward_fn,
    )


def build_policy(env, policy_cfg: dict):
    return Policy(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(policy_cfg["hidden_dim"]),
        n_hidden_layers=int(policy_cfg["n_hidden_layers"]),
    )


def build_reward(env, reward_cfg: dict):
    return Reward(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(reward_cfg["hidden_dim"]),
        n_hidden_layers=int(reward_cfg["n_hidden_layers"]),
        clamp_magnitude=float(reward_cfg["clamp_magnitude"]),
    )
