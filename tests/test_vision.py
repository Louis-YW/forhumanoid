from __future__ import annotations

import numpy as np

from trace2touch.board import render_board_array
from trace2touch.vision import detect_markers, estimate_homography, transform_points


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
