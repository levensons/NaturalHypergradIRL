import importlib
from types import ModuleType

from src.algorithms.reinforce import REINFORCE
from src.algorithms.sac import SAC


SUPPORTED_ENVS = {"cartpole", "hopper", "lqr"}
SUPPORTED_AGENTS = {"reinforce", "sac"}

def import_env_builders(env_name: str) -> ModuleType:
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported environment: {env_name}. Available: {sorted(SUPPORTED_ENVS)}")

    module_path = f"src.envs.{env_name}.builders"

    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        if error.name == module_path:
            raise ModuleNotFoundError(f"Environment builders module does not exist: {module_path}") from error
        raise


def build_agent(policy, env, agent_cfg: dict, gamma: float, alpha: float):
    agent_type = agent_cfg["type"]
    params = agent_cfg["params"]

    if agent_type == "sac":
        return SAC(
            policy=policy,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=int(params["q_hidden_dim"]),
            n_hidden_layers=int(params["q_n_hidden_layers"]),
            gamma=float(gamma),
            alpha=float(alpha),
            tau=float(params["tau"]),
            replay_buffer_capacity=int(params["replay_buffer_capacity"]),
        )

    if agent_type == "reinforce":
        return REINFORCE(
            policy=policy,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            gamma=float(gamma),
            alpha=float(alpha),
            use_baseline=bool(agent_cfg.get("use_baseline", False)),
            baseline_momentum=agent_cfg.get("baseline_momentum"),
        )

    raise ValueError(f"Unsupported agent: {agent_type}. Available: {sorted(SUPPORTED_AGENTS)}")


def build_arch(
    env,
    env_cfg: dict,
    policy_cfg: dict,
    reward_cfg: dict,
    method: str,
    agent_type: str,
) -> dict:
    return {
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "method": method,
        "agent": agent_type,
        "env": dict(env_cfg),
        "policy": dict(policy_cfg),
        "reward": dict(reward_cfg),
    }
