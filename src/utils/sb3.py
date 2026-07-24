from pathlib import Path

import torch.nn as nn


def normalize_sb3_load_path(path: str | Path) -> Path:
    path = Path(path)

    if path.suffix != ".zip":
        path = path.with_suffix(".zip")

    return path


def normalize_sb3_save_path(path: str | Path) -> Path:
    path = Path(path)

    if path.suffix == ".zip":
        path = path.with_suffix("")

    return path


def init_sb3_model(algo: str, env, params: dict, verbose: int = 1):
    from stable_baselines3 import PPO, SAC

    algo = algo.lower()
    params = dict(params)

    policy = params.pop("policy", "MlpPolicy")

    if algo == "sac":
        return SAC(policy, env, **params, verbose=verbose)

    if algo == "ppo":
        return PPO(policy, env, **params, verbose=verbose)

    raise ValueError(f"Unknown SB3 algorithm: {algo}")


def load_sb3_model(algo: str, path: str | Path):
    from stable_baselines3 import PPO, SAC

    algo = algo.lower()
    path = normalize_sb3_load_path(path)

    if not path.exists():
        raise FileNotFoundError(f"SB3 checkpoint not found: {path}")

    if algo == "sac":
        return SAC.load(str(path))

    if algo == "ppo":
        return PPO.load(str(path))

    raise ValueError(f"Unknown SB3 algorithm: {algo}")


def init_policy_from_sb3_sac_expert(policy: nn.Module, expert_path: str | Path) -> None:
    expert = load_sb3_model("sac", expert_path)
    actor_state = expert.actor.state_dict()

    activation = getattr(policy, "activation", None)
    expected_activation_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}.get(activation)
    if expected_activation_cls is None:
        raise ValueError(
            "Cannot initialize policy from SB3 expert; unsupported "
            f"policy.activation={activation}."
        )

    actor_activations = [
        layer
        for layer in expert.actor.latent_pi
        if not isinstance(layer, nn.Linear)
    ]
    if any(not isinstance(layer, expected_activation_cls) for layer in actor_activations):
        raise ValueError(
            "Cannot initialize policy from SB3 expert; actor activations do not match "
            f"policy.activation={activation}."
        )

    hidden_layers = [
        (name, layer)
        for name, layer in policy.backbone.named_children()
        if isinstance(layer, nn.Linear)
    ]

    mappings = []
    for layer_idx, (name, layer) in enumerate(hidden_layers):
        sb3_idx = 2 * layer_idx
        mappings.extend(
            [
                (f"backbone.{name}.weight", f"latent_pi.{sb3_idx}.weight"),
                (f"backbone.{name}.bias", f"latent_pi.{sb3_idx}.bias"),
            ]
        )

    mappings.extend(
        [
            ("mean_head.weight", "mu.weight"),
            ("mean_head.bias", "mu.bias"),
            ("log_std_head.weight", "log_std.weight"),
            ("log_std_head.bias", "log_std.bias"),
        ]
    )

    policy_state = policy.state_dict()
    missing_keys = [source_key for _, source_key in mappings if source_key not in actor_state]
    if missing_keys:
        raise ValueError(f"SB3 expert actor is missing keys: {missing_keys}")

    shape_mismatches = []
    for target_key, source_key in mappings:
        target = policy_state[target_key]
        source = actor_state[source_key]
        if tuple(source.shape) != tuple(target.shape):
            shape_mismatches.append((target_key, tuple(source.shape), tuple(target.shape)))

    if shape_mismatches:
        details = ", ".join(
            f"{key}: expert{source_shape} != policy{target_shape}"
            for key, source_shape, target_shape in shape_mismatches
        )
        raise ValueError(f"Cannot initialize policy from SB3 expert; shape mismatch: {details}")

    for target_key, source_key in mappings:
        source = actor_state[source_key]
        target = policy_state[target_key]
        target.copy_(source.to(device=target.device, dtype=target.dtype))

    policy.load_state_dict(policy_state)
