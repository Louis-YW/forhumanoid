from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from .trajectory import Demonstration, sha256_file


def detect_markers(frame_bgr: np.ndarray) -> dict[int, np.ndarray]:
    """Return ArUco marker centres indexed by ID."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # OpenCV 4.6 compatibility
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return {}
    return {
        int(marker_id): np.asarray(corner, dtype=float).reshape(4, 2).mean(axis=0)
        for marker_id, corner in zip(ids.ravel(), corners)
    }


def estimate_homography(
    observed_px: dict[int, np.ndarray], marker_world_m: dict[int, tuple[float, float]]
) -> np.ndarray:
    required = sorted(marker_world_m)
    missing = [marker_id for marker_id in required if marker_id not in observed_px]
    if missing:
        raise ValueError(f"calibration markers not detected: {missing}")
    source = np.float32([observed_px[marker_id] for marker_id in required])
    target = np.float32([marker_world_m[marker_id] for marker_id in required])
    homography, status = cv2.findHomography(source, target, method=0)
    if homography is None or status is None:
        raise ValueError("unable to estimate board homography")
    return homography


def transform_points(points_px: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2).astype(float)


def detect_coloured_object(
    frame_bgr: np.ndarray,
    hsv_ranges: list[dict[str, list[int]]],
    *,
    min_area_px: float,
    max_area_fraction: float = 0.08,
) -> tuple[np.ndarray | None, float, np.ndarray]:
    """Find the largest colour blob and return centroid, confidence and mask."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for bounds in hsv_ranges:
        lower = np.asarray(bounds["lower"], dtype=np.uint8)
        upper = np.asarray(bounds["upper"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = frame_bgr.shape[0] * frame_bgr.shape[1] * max_area_fraction
    contours = [
        contour for contour in contours if min_area_px <= cv2.contourArea(contour) <= max_area
    ]
    if not contours:
        return None, 0.0, mask
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] <= 1e-6:
        return None, 0.0, mask
    centroid = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=float,
    )
    area = cv2.contourArea(contour)
    perimeter = max(cv2.arcLength(contour, True), 1e-6)
    circularity = float(np.clip(4 * np.pi * area / perimeter**2, 0.0, 1.0))
    area_score = float(np.clip(area / (4.0 * min_area_px), 0.0, 1.0))
    confidence = 0.35 * area_score + 0.65 * circularity
    return centroid, confidence, mask


def _select_calibrated_object(
    frame_bgr: np.ndarray,
    homography: np.ndarray,
    hsv_ranges: list[dict[str, list[int]]],
    *,
    min_area_px: float,
    max_area_fraction: float,
    object_radius_m: float,
    previous_m: np.ndarray | None,
    board_size_m: tuple[float, float],
    min_area_ratio: float,
    max_area_ratio: float,
    min_circularity: float,
    max_step_m: float,
    board_margin_m: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float, np.ndarray]:
    """Select the colour blob that is physically consistent with the known object.

    The largest-blob heuristic is useful for a clean synthetic scene, but it can
    jump to a similarly coloured sleeve. Metric size, board bounds and temporal
    continuity make the choice auditable and independent of image resolution.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for bounds in hsv_ranges:
        lower = np.asarray(bounds["lower"], dtype=np.uint8)
        upper = np.asarray(bounds["upper"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area_px = frame_bgr.shape[0] * frame_bgr.shape[1] * max_area_fraction
    expected_area_m2 = np.pi * object_radius_m**2
    width_m, height_m = board_size_m
    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        if not min_area_px <= area_px <= max_area_px:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        centre_px = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=float,
        )
        centre_m = transform_points(centre_px[None, :], homography)[0]
        if not (
            -board_margin_m <= centre_m[0] <= width_m + board_margin_m
            and -board_margin_m <= centre_m[1] <= height_m + board_margin_m
        ):
            continue
        contour_m = transform_points(contour.reshape(-1, 2), homography).astype(np.float32)
        area_m2 = abs(float(cv2.contourArea(contour_m.reshape(-1, 1, 2))))
        area_ratio = area_m2 / max(expected_area_m2, 1e-12)
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            continue
        perimeter_px = max(float(cv2.arcLength(contour, True)), 1e-6)
        circularity = float(np.clip(4 * np.pi * area_px / perimeter_px**2, 0.0, 1.0))
        if circularity < min_circularity:
            continue
        distance_m = 0.0 if previous_m is None else float(np.linalg.norm(centre_m - previous_m))
        if previous_m is not None and distance_m > max_step_m:
            continue

        size_score = float(np.exp(-abs(np.log(max(area_ratio, 1e-6)))))
        continuity_score = (
            1.0 if previous_m is None else float(np.exp(-distance_m / max(max_step_m, 1e-6)))
        )
        confidence = float(np.clip(0.45 * size_score + 0.35 * circularity + 0.20 * continuity_score, 0, 1))
        score = confidence + (0.30 * continuity_score if previous_m is not None else 0.0)
        candidates.append((score, centre_px, centre_m, confidence))

    if not candidates:
        return None, None, 0.0, mask
    _, centre_px, centre_m, confidence = max(candidates, key=lambda item: item[0])
    return centre_px, centre_m, confidence, mask


def _longest_false_run(valid: np.ndarray) -> int:
    longest = current = 0
    for value in valid:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _interpolate_and_smooth(
    time_s: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    window: int,
) -> np.ndarray:
    if valid.sum() < 4:
        raise ValueError("fewer than four valid object detections")
    filled = np.empty_like(points, dtype=float)
    for axis in range(2):
        filled[:, axis] = np.interp(time_s, time_s[valid], points[valid, axis])
    max_window = len(filled) if len(filled) % 2 else len(filled) - 1
    window = min(max(window, 5), max_window)
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        filled = savgol_filter(filled, window_length=window, polyorder=2, axis=0)
    return filled


def _motion_crop(
    time_s: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    threshold_m_s: float,
    pad_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.linalg.norm(np.gradient(points, time_s, axis=0), axis=1)
    moving = np.flatnonzero(velocity >= threshold_m_s)
    if len(moving) < 3:
        return time_s, points, confidence
    start = max(0, int(moving[0]) - pad_frames)
    stop = min(len(time_s), int(moving[-1]) + pad_frames + 1)
    return time_s[start:stop], points[start:stop], confidence[start:stop]


def _resample(
    time_s: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_t = np.linspace(float(time_s[0]), float(time_s[-1]), n_samples)
    target_points = np.column_stack(
        [PchipInterpolator(time_s, points[:, axis])(target_t) for axis in range(2)]
    )
    target_confidence = np.clip(np.interp(target_t, time_s, confidence), 0.0, 1.0)
    target_t -= target_t[0]
    return target_t, target_points, target_confidence


def extract_demonstration(
    video_path: str | Path,
    config: dict[str, Any],
    *,
    overlay_path: str | Path | None = None,
) -> Demonstration:
    """Extract, calibrate and smooth a tabletop object trajectory from video."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    calibration_frames = int(config["vision"].get("calibration_frames", 20))
    marker_samples: dict[int, list[np.ndarray]] = defaultdict(list)
    for _ in range(calibration_frames):
        ok, frame = cap.read()
        if not ok:
            break
        for marker_id, centre in detect_markers(frame).items():
            marker_samples[marker_id].append(centre)
    marker_world = {
        int(marker_id): tuple(map(float, xy))
        for marker_id, xy in config["vision"]["marker_world_m"].items()
    }
    observed = {
        marker_id: np.median(np.stack(samples), axis=0)
        for marker_id, samples in marker_samples.items()
        if samples
    }
    homography = estimate_homography(observed, marker_world)
    marker_width = max(x for x, _ in marker_world.values()) - min(
        x for x, _ in marker_world.values()
    )
    marker_height = max(y for _, y in marker_world.values()) - min(
        y for _, y in marker_world.values()
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    points_m_raw: list[np.ndarray] = []
    confidences: list[float] = []
    overlays: list[np.ndarray] = []
    per_frame_calibrations = 0
    previous_m: np.ndarray | None = None
    frame_index = 0
    overlay_stride = max(1, int(config["vision"].get("overlay_stride", 2)))
    object_radius_m = float(config["vision"].get("object_radius_m", 0.0))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_markers = detect_markers(frame)
        if set(marker_world).issubset(frame_markers):
            frame_homography = estimate_homography(frame_markers, marker_world)
            per_frame_calibrations += 1
        else:
            frame_homography = homography

        if object_radius_m > 0:
            centroid, point_m, confidence, _ = _select_calibrated_object(
                frame,
                frame_homography,
                config["vision"]["object_hsv_ranges"],
                min_area_px=float(config["vision"].get("min_area_px", 80)),
                max_area_fraction=float(config["vision"].get("max_area_fraction", 0.08)),
                object_radius_m=object_radius_m,
                previous_m=previous_m,
                board_size_m=(marker_width, marker_height),
                min_area_ratio=float(config["vision"].get("min_area_ratio", 0.25)),
                max_area_ratio=float(config["vision"].get("max_area_ratio", 2.0)),
                min_circularity=float(config["vision"].get("min_circularity", 0.20)),
                max_step_m=float(config["vision"].get("max_step_m", 0.020)),
                board_margin_m=float(config["vision"].get("board_margin_m", 0.015)),
            )
        else:
            centroid, confidence, _ = detect_coloured_object(
                frame,
                config["vision"]["object_hsv_ranges"],
                min_area_px=float(config["vision"].get("min_area_px", 80)),
                max_area_fraction=float(config["vision"].get("max_area_fraction", 0.08)),
            )
            point_m = (
                transform_points(centroid[None, :], frame_homography)[0]
                if centroid is not None
                else None
            )
        if point_m is not None:
            previous_m = point_m
        points_m_raw.append(
            point_m if point_m is not None else np.array([np.nan, np.nan], dtype=float)
        )
        confidences.append(confidence)
        if overlay_path is not None and frame_index % overlay_stride == 0:
            draw = frame.copy()
            if centroid is not None:
                cv2.circle(draw, tuple(np.round(centroid).astype(int)), 10, (0, 255, 0), 3)
            for marker_id, centre in frame_markers.items():
                pt = tuple(np.round(centre).astype(int))
                cv2.circle(draw, pt, 5, (255, 150, 0), -1)
                cv2.putText(
                    draw, str(marker_id), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2
                )
            max_width = int(config["vision"].get("overlay_max_width_px", draw.shape[1]))
            if max_width > 0 and draw.shape[1] > max_width:
                scale = max_width / draw.shape[1]
                draw = cv2.resize(
                    draw,
                    (max_width, round(draw.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            overlays.append(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
        frame_index += 1
    cap.release()
    if frame_index < 4:
        raise ValueError("video contains fewer than four readable frames")

    points_m = np.asarray(points_m_raw, dtype=float)
    confidence = np.asarray(confidences, dtype=float)
    valid = np.isfinite(points_m).all(axis=1)
    time_s = np.arange(frame_index, dtype=float) / fps
    points_m = _interpolate_and_smooth(
        time_s,
        points_m,
        valid,
        window=int(config["vision"].get("smoothing_window", 11)),
    )
    time_s, points_m, confidence = _motion_crop(
        time_s,
        points_m,
        confidence,
        threshold_m_s=float(config["vision"].get("motion_threshold_m_s", 0.004)),
        pad_frames=int(config["vision"].get("motion_pad_frames", 5)),
    )
    time_s, points_m, confidence = _resample(
        time_s,
        points_m,
        confidence,
        n_samples=int(config["trajectory"].get("samples", 80)),
    )

    if overlay_path is not None and overlays:
        overlay_path = Path(overlay_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(overlay_path, overlays, fps=max(1.0, fps / overlay_stride), loop=0)

    provenance = config.get("provenance", {})
    config_snapshot = {key: value for key, value in config.items() if key != "_config_path"}
    demo = Demonstration(
        time_s=time_s,
        object_xy_m=points_m,
        confidence=confidence,
        workspace_size_m=(float(marker_width), float(marker_height)),
        source_sha256=sha256_file(video_path),
        metadata={
            "source_filename": video_path.name,
            "fps": fps,
            "frames": frame_index,
            "valid_detection_fraction": float(valid.mean()),
            "longest_interpolated_gap_frames": _longest_false_run(valid),
            "per_frame_calibration_fraction": float(per_frame_calibrations / frame_index),
            "marker_centres_px": {str(k): v.tolist() for k, v in observed.items()},
            "homography_px_to_m": homography.tolist(),
            "collected_by": provenance.get("collected_by", "unspecified"),
            "collection_date": str(provenance.get("collection_date", "unspecified")),
            "synthetic": bool(provenance.get("synthetic", False)),
            "notes": provenance.get("notes", ""),
            "config_snapshot": json.loads(json.dumps(config_snapshot, default=str)),
        },
    )
    demo.validate()
    return demo
