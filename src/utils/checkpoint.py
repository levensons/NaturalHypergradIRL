from pathlib import Path
import torch


def save_checkpoint(path: str, policy, reward, arch: dict, **extra) -> None:
    checkpoint = {
        "format_version": 1,
        "arch": arch,
        "policy_state_dict": policy.state_dict(),
        "reward_state_dict": reward.state_dict(),
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "arch" not in ckpt:
        ckpt["arch"] = {
            "state_dim": int(ckpt.get("state_dim", 0)),
            "action_dim": int(ckpt.get("action_dim", 0)),
            "policy_hidden": 64,
            "policy_n_hidden_layers": 2,
            "reward_hidden": 64,
            "reward_gamma": 0.99,
        }

    return ckpt
