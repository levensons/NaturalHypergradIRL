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
    deterministic: bool = False,
    device: torch.device | str = None,
) -> dict:
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(policy.parameters()).device

    was_training = policy.training
    policy.eval()

    video_env = RecordVideo(
        env=env.clone().env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=name_prefix,
        disable_logger=True,
    )

    total_reward = 0.0
    steps = 0

    state, _ = video_env.reset()

    while True:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
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

    generated_files = sorted(video_dir.glob(f"{name_prefix}-episode-*.mp4"), key=lambda path: path.stat().st_mtime)

    if not generated_files:
        raise RuntimeError("RecordVideo did not create a video file.")

    generated_path = generated_files[-1]

    video_path = video_dir / f"{name_prefix}.mp4"

    if video_path.exists():
        video_path.unlink()

    generated_path.rename(video_path)

    stats = {
        "return": total_reward,
        "length": steps,
        "video_path": video_path,
    }

    print(f"[video] return={total_reward:.2f} " f"steps={steps} " f"path={video_path}")

    return stats
