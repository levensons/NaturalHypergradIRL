from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(env_name: str | None, config_path: str | None) -> str:
    if config_path is not None:
        return config_path

    if env_name is None:
        raise ValueError("Specify either --env or --config.")

    return f"config/{env_name}.yaml"
