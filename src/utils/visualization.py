from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mlflow
import numpy as np
import torch
from PIL import Image

from src.utils.mlflow import log_artifact_if_exists


def _write_gif_gallery(
    *,
    output_path: Path,
    title: str,
    gif_names: list[str],
    stats: list[dict[str, float]],
) -> None:
    rows = []
    for idx, (gif_name, traj_stats) in enumerate(zip(gif_names, stats)):
        rows.append(
            "\n".join(
                [
                    "<section>",
                    f"<h2>Trajectory {idx}</h2>",
                    f"<img src=\"{gif_name}\" alt=\"Trajectory {idx}\" />",
                    (
                        "<p>"
                        f"return={traj_stats['return']:.3f}, "
                        f"length={traj_stats['length']:.0f}, "
                        f"frames={traj_stats['frames']:.0f}"
                        "</p>"
                    ),
                    "</section>",
                ]
            )
        )

    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "<meta charset=\"utf-8\" />",
            f"<title>{title}</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 24px; background: #111; color: #eee; }",
            "main { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; }",
            "section { border: 1px solid #333; padding: 16px; background: #181818; }",
            "img { width: 100%; height: auto; display: block; }",
            "p { color: #bbb; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            "<main>",
            *rows,
            "</main>",
            "</body>",
            "</html>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def render_policy_trajectory_gif(
    *,
    env_id: str,
    policy,
    seed: int,
    max_steps: int,
    deterministic: bool,
    frame_stride: int,
    fps: int,
    output_path: Path,
) -> dict[str, float]:
    frame_stride = max(1, int(frame_stride))
    fps = max(1, int(fps))

    env = gym.make(env_id, render_mode="rgb_array")
    was_training = policy.training

    try:
        state, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        frames = []
        total_return = 0.0
        trajectory_len = 0
        device = next(policy.parameters()).device

        policy.eval()

        for step in range(max_steps):
            if step % frame_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(np.asarray(frame)))

            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).flatten()

            with torch.no_grad():
                action = policy.sample(state_tensor, deterministic=deterministic).detach().cpu()

            action_for_env = np.asarray(action.numpy(), dtype=np.float32).reshape(env.action_space.shape)
            state, reward, terminated, truncated, _ = env.step(action_for_env)

            total_return += float(reward)
            trajectory_len = step + 1

            if terminated or truncated:
                break

        if frames:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )

        return {
            "return": total_return,
            "length": float(trajectory_len),
            "frames": float(len(frames)),
        }
    finally:
        if was_training:
            policy.train()
        else:
            policy.eval()
        env.close()


def log_policy_trajectory_gifs(
    *,
    env_id: str,
    policy,
    outer_step: int,
    max_steps: int,
    output_root: Path,
    cfg: dict[str, Any] | None = None,
    seed_base: int = 0,
    logger=None,
) -> bool:
    cfg = cfg or {}
    if not bool(cfg.get("enabled", True)):
        return True

    n_traj = int(cfg.get("n_traj", 3))
    if n_traj <= 0:
        return True

    max_steps = int(cfg.get("max_steps", max_steps))
    frame_stride = int(cfg.get("frame_stride", 4))
    fps = int(cfg.get("fps", 20))
    deterministic = bool(cfg.get("deterministic", True))
    seed_base = int(cfg.get("seed", seed_base))
    artifact_dir = str(cfg.get("artifact_dir", "policy_trajectories"))
    local_dir = output_root / artifact_dir / f"before_outer_update_{outer_step:04d}"

    stats = []
    gif_names = []
    try:
        for traj_idx in range(n_traj):
            gif_name = f"traj_{traj_idx:02d}.gif"
            output_path = local_dir / gif_name
            traj_stats = render_policy_trajectory_gif(
                env_id=env_id,
                policy=policy,
                seed=seed_base + outer_step * n_traj + traj_idx,
                max_steps=max_steps,
                deterministic=deterministic,
                frame_stride=frame_stride,
                fps=fps,
                output_path=output_path,
            )
            stats.append(traj_stats)
            gif_names.append(gif_name)
            log_artifact_if_exists(
                output_path,
                artifact_path=f"{artifact_dir}/before_outer_update_{outer_step:04d}",
            )
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Policy trajectory visualization disabled after failure: {exc}")
        return False

    if not stats:
        return True

    gallery_path = local_dir / "index.html"
    _write_gif_gallery(
        output_path=gallery_path,
        title=f"Policy Trajectories Before Outer Update {outer_step}",
        gif_names=gif_names,
        stats=stats,
    )
    log_artifact_if_exists(
        gallery_path,
        artifact_path=f"{artifact_dir}/before_outer_update_{outer_step:04d}",
    )

    returns = [item["return"] for item in stats]
    lengths = [item["length"] for item in stats]
    frames = [item["frames"] for item in stats]

    if logger is not None:
        logger.info(
            f"visualized {len(stats)} policy trajectories before outer update {outer_step}: "
            f"return={float(np.mean(returns)):.1f}, "
            f"len={float(np.mean(lengths)):.1f}, "
            f"frames={float(np.mean(frames)):.1f}"
        )

    if mlflow.active_run() is not None:
        mlflow.log_metrics(
            {
                "policy_viz/return": float(np.mean(returns)),
                "policy_viz/length": float(np.mean(lengths)),
                "policy_viz/frames": float(np.mean(frames)),
            },
            step=outer_step,
        )

    return True
