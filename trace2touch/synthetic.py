from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

from .board import A4_LANDSCAPE_M, render_board_array


def _bezier(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = control
    u = 1.0 - t
    return (
        u[:, None] ** 3 * p0
        + 3 * u[:, None] ** 2 * t[:, None] * p1
        + 3 * u[:, None] * t[:, None] ** 2 * p2
        + t[:, None] ** 3 * p3
    )


def make_synthetic_video(
    output: str | Path,
    config: dict[str, Any],
    *,
    seed: int = 7,
) -> Path:
    """Create a deterministic video used only to test the full software path."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    board = render_board_array(1188)
    bh, bw = board.shape[:2]
    fps = int(config.get("synthetic", {}).get("fps", 30))
    n_frames = int(config.get("synthetic", {}).get("frames", 120))

    # Perspective warp emulates a phone that is close to, but not exactly, top-down.
    destination = np.float32([[55, 52], [906, 24], [934, 691], [24, 666]])
    source = np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]])
    warp = cv2.getPerspectiveTransform(source, destination)
    width_m, height_m = A4_LANDSCAPE_M

    start = np.array([0.068, 0.142])
    goal = np.array([0.235, 0.058])
    # The curved human path goes above the central obstacle. A direct start-to-goal
    # controller collides, making the ablation meaningful instead of decorative.
    control = np.array([start, [0.078, 0.012], [0.195, 0.012], goal], dtype=float)
    phase = np.clip((np.linspace(0, 1, n_frames) - 0.10) / 0.80, 0, 1)
    phase = phase * phase * (3 - 2 * phase)
    trajectory = _bezier(control, phase)
    trajectory += rng.normal(0, 0.00055, size=trajectory.shape)

    obstacle = config.get("simulation", {}).get("obstacles", [{}])[0]
    obstacle_board = np.asarray(obstacle.get("board_xy_m", [0.148, 0.102]), dtype=float)
    obstacle_radius_m = float(obstacle.get("board_radius_m", 0.028))

    def board_to_px(xy_m: np.ndarray) -> tuple[int, int]:
        return round(float(xy_m[0] / width_m * bw)), round(float(xy_m[1] / height_m * bh))

    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
    try:
        for frame_index, xy_m in enumerate(trajectory):
            frame = board.copy()
            obstacle_px = board_to_px(obstacle_board)
            radius_px = max(4, round(obstacle_radius_m / width_m * bw))
            cv2.circle(frame, obstacle_px, radius_px, (95, 95, 95), -1, cv2.LINE_AA)
            cv2.circle(frame, obstacle_px, radius_px, (30, 30, 30), 3, cv2.LINE_AA)

            puck_px = board_to_px(xy_m)
            cv2.circle(frame, puck_px, 26, (210, 35, 210), -1, cv2.LINE_AA)
            cv2.circle(frame, puck_px, 26, (80, 10, 80), 3, cv2.LINE_AA)
            if frame_index > 0:
                trail = trajectory[max(0, frame_index - 20) : frame_index]
                trail_px = np.asarray([board_to_px(point) for point in trail], dtype=np.int32)
                if len(trail_px) > 1:
                    cv2.polylines(frame, [trail_px], False, (230, 130, 230), 4, cv2.LINE_AA)

            warped = cv2.warpPerspective(frame, warp, (960, 720), borderValue=(32, 38, 46))
            writer.append_data(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    return output
