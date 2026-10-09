from __future__ import annotations

import cv2
import numpy as np

from trace2touch.board import render_board_array
from trace2touch.vision import (
    _select_calibrated_object,
    detect_markers,
    estimate_homography,
    transform_points,
)


def test_homography_recovers_metric_coordinates() -> None:
    pixel = {
        0: np.array([100.0, 80.0]),
        1: np.array([900.0, 60.0]),
        2: np.array([940.0, 650.0]),
        3: np.array([70.0, 680.0]),
    }
    world = {0: (0.0, 0.0), 1: (0.247, 0.0), 2: (0.247, 0.160), 3: (0.0, 0.160)}
    homography = estimate_homography(pixel, world)
    recovered = transform_points(np.stack([pixel[i] for i in range(4)]), homography)
    expected = np.asarray([world[i] for i in range(4)])
    np.testing.assert_allclose(recovered, expected, atol=1e-5)


def test_generated_board_contains_all_calibration_markers() -> None:
    markers = detect_markers(render_board_array(900))
    assert set(markers) == {0, 1, 2, 3}


def test_metric_tracker_rejects_larger_colour_distractor() -> None:
    frame = np.full((400, 400, 3), 240, dtype=np.uint8)
    cv2.circle(frame, (100, 100), 25, (140, 20, 150), -1)
    cv2.circle(frame, (300, 300), 48, (140, 20, 150), -1)
    homography = np.array([[0.0005, 0.0, 0.0], [0.0, 0.0005, 0.0], [0.0, 0.0, 1.0]])
    centre_px, centre_m, confidence, _ = _select_calibrated_object(
        frame,
        homography,
        [{"lower": [135, 25, 30], "upper": [179, 255, 255]}],
        min_area_px=100,
        max_area_fraction=0.10,
        object_radius_m=0.0125,
        previous_m=np.array([0.05, 0.05]),
        board_size_m=(0.20, 0.20),
        min_area_ratio=0.5,
        max_area_ratio=1.5,
        min_circularity=0.7,
        max_step_m=0.02,
        board_margin_m=0.0,
    )
    np.testing.assert_allclose(centre_px, [100, 100], atol=1.0)
    np.testing.assert_allclose(centre_m, [0.05, 0.05], atol=1e-3)
    assert confidence > 0.8
