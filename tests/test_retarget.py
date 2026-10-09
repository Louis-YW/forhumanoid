from __future__ import annotations

import numpy as np

from trace2touch.retarget import retarget_demonstration
from trace2touch.trajectory import Demonstration


def test_retarget_adds_approach_and_follow_through() -> None:
    time = np.linspace(0.0, 1.0, 20)
    path = np.column_stack([0.04 + 0.12 * time, 0.12 - 0.04 * np.sin(np.pi * time)])
    demo = Demonstration(
        time_s=time,
        object_xy_m=path,
        confidence=np.ones(20),
        workspace_size_m=(0.247, 0.160),
        source_sha256="b" * 64,
    )
    config = {
        "retarget": {
            "target_span_m": 0.3,
            "min_scale": 0.5,
            "max_scale": 4.0,
            "sim_center_xy": [0.08, 0.0],
            "approach_samples": 5,
            "approach_distance_m": 0.06,
            "follow_through_samples": 6,
            "follow_through_distance_m": 0.08,
            "contact_overlap_m": 0.01,
        },
        "simulation": {"puck_radius_m": 0.038, "ee_radius_m": 0.035, "obstacles": []},
    }
    plan = retarget_demonstration(demo, config)
    assert len(plan.ee_xy) == len(path) + 11
    np.testing.assert_allclose(plan.object_xy[:5], np.repeat(plan.start_xy[None, :], 5, axis=0))
    np.testing.assert_allclose(plan.object_xy[-6:], np.repeat(plan.goal_xy[None, :], 6, axis=0))
    assert np.linalg.norm(plan.ee_xy[-1] - plan.ee_xy[-7]) > 0.07
