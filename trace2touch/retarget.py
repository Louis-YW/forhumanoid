from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from .trajectory import Demonstration, RetargetedPlan


def _unit_tangents(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    max_window = len(path) if len(path) % 2 else len(path) - 1
    window = min(11, max_window)
    smoothed = savgol_filter(path, window, 2, axis=0) if window >= 5 else path
    radius = min(3, max(1, len(path) // 10))
    delta = np.empty_like(smoothed)
    for index in range(len(path)):
        left = max(0, index - radius)
        right = min(len(path) - 1, index + radius)
        delta[index] = smoothed[right] - smoothed[left]
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    motion_floor = max(1e-5, float(norm.max()) * 0.03)
    tangent = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > motion_floor)
    valid = np.flatnonzero(norm[:, 0] > motion_floor)
    if not len(valid):
        raise ValueError("the recorded object does not move far enough to infer a push direction")
    for index in range(len(tangent)):
        nearest = valid[np.argmin(np.abs(valid - index))]
        if np.linalg.norm(tangent[index]) <= 0.5:
            tangent[index] = tangent[nearest]
    if window >= 5:
        tangent = savgol_filter(tangent, window, 2, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
    return tangent


def retarget_demonstration(demo: Demonstration, config: dict[str, Any]) -> RetargetedPlan:
    """Map a calibrated human object path to a robot end-effector contact path."""
    options = config["retarget"]
    source = np.asarray(demo.object_xy_m, dtype=float)
    source_span = float(max(np.ptp(source[:, 0]), np.ptp(source[:, 1]), 1e-6))
    target_span = float(options.get("target_span_m", 0.32))
    scale = target_span / source_span
    scale = float(np.clip(scale, options.get("min_scale", 0.5), options.get("max_scale", 4.0)))

    source_anchor = (source[0] + source[-1]) / 2.0
    centred = source - source_anchor
    centred[:, 1] *= -1.0  # image/board y points down; MuJoCo y points left.
    angle = np.deg2rad(float(options.get("rotation_deg", 0.0)))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
    )
    target_centre = np.asarray(options.get("sim_center_xy", [0.08, 0.0]), dtype=float)

    def transform(points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points).astype(float)
        shifted = points - source_anchor
        shifted[:, 1] *= -1.0
        return shifted @ rotation.T * scale + target_centre

    object_xy = transform(source)
    tangent = _unit_tangents(object_xy)
    puck_radius = float(config["simulation"].get("puck_radius_m", 0.038))
    ee_radius = float(config["simulation"].get("ee_radius_m", 0.035))
    overlap = float(options.get("contact_overlap_m", 0.006))
    contact_distance = puck_radius + ee_radius - overlap
    contact_xy = object_xy - contact_distance * tangent

    approach_samples = int(options.get("approach_samples", 10))
    approach_distance = float(options.get("approach_distance_m", 0.07))
    if approach_samples > 0:
        start_far = contact_xy[0] - approach_distance * tangent[0]
        alpha = np.linspace(0.0, 1.0, approach_samples, endpoint=False)[:, None]
        approach = start_far[None, :] * (1.0 - alpha) + contact_xy[0][None, :] * alpha
        ee_xy = np.vstack([approach, contact_xy])
        object_xy = np.vstack([np.repeat(object_xy[:1], approach_samples, axis=0), object_xy])
        confidence = np.concatenate(
            [np.repeat(demo.confidence[:1], approach_samples), demo.confidence]
        )
    else:
        ee_xy = contact_xy
        confidence = demo.confidence.copy()

    follow_samples = int(options.get("follow_through_samples", 8))
    follow_distance = float(options.get("follow_through_distance_m", 0.065))
    if follow_samples > 0:
        alpha = np.linspace(0.0, 1.0, follow_samples + 1, endpoint=True)[1:, None]
        follow = contact_xy[-1][None, :] + alpha * follow_distance * tangent[-1]
        ee_xy = np.vstack([ee_xy, follow])
        object_xy = np.vstack([object_xy, np.repeat(object_xy[-1:], follow_samples, axis=0)])
        confidence = np.concatenate([confidence, np.repeat(demo.confidence[-1:], follow_samples)])

    obstacles: list[dict[str, float]] = []
    for obstacle in config.get("simulation", {}).get("obstacles", []):
        if "board_xy_m" in obstacle:
            xy = transform(np.asarray(obstacle["board_xy_m"], dtype=float))[0]
            radius = float(obstacle.get("board_radius_m", 0.02)) * scale
        else:
            xy = np.asarray(obstacle["sim_xy_m"], dtype=float)
            radius = float(obstacle["sim_radius_m"])
        obstacles.append(
            {
                "x": float(xy[0]),
                "y": float(xy[1]),
                "radius": float(radius),
                "height": float(obstacle.get("height_m", 0.09)),
            }
        )

    return RetargetedPlan(
        object_xy=object_xy,
        ee_xy=ee_xy,
        confidence=np.clip(confidence, 0.0, 1.0),
        start_xy=object_xy[0].copy(),
        goal_xy=object_xy[-1].copy(),
        obstacles=obstacles,
        transform={
            "source_anchor_m": source_anchor.tolist(),
            "scale": scale,
            "rotation_deg": float(options.get("rotation_deg", 0.0)),
            "sim_center_xy": target_centre.tolist(),
            "contact_distance_m": contact_distance,
        },
    )
