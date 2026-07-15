from pathlib import Path

import torch
from gymnasium.wrappers import RecordVideo

from src.utils.env import Environment
from src.utils.policies import Policy


@torch.no_grad()
def record_policy_video(
    env: Environment,
    policy: Policy,
    video_dir: str | Path,
    name_prefix: str = "policy",
    max_steps: int = 1000,
    deterministic: bool = False,
    device: torch.device | str | None = None,
) -> dict:
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(policy.parameters()).device

    was_training = policy.training
    policy.eval()

    video_env = RecordVideo(
        env=env.env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=name_prefix,
        disable_logger=True,
    )

    total_reward = 0.0
    steps = 0

    state, _ = video_env.reset()

    for _ in range(max_steps):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        action = policy.sample(states=state_tensor, deterministic=deterministic)
        action = action.squeeze(0).detach().cpu().numpy()

        state, reward, terminated, truncated, _ = video_env.step(action)

        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            break

    video_env.close()

    if was_training:
        policy.train()

    video_files = sorted(video_dir.glob(f"{name_prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
    video_path = str(video_files[-1]) if video_files else None

    stats = {
        "return": total_reward,
        "length": steps,
        "video_path": video_path,
    }

    print(f"[video] return={total_reward:.2f} " f"steps={steps} " f"path={video_path}")

    return stats
