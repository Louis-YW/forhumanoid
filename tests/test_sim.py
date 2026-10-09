from __future__ import annotations

import numpy as np

from trace2touch.sim import PlanarArmEnv


def test_inverse_kinematics_reaches_tabletop_points() -> None:
    points = np.array([[-0.05, 0.15], [0.10, -0.12], [0.24, 0.04]])
    q = None
    for target in points:
        q = PlanarArmEnv.inverse_kinematics(target, q)
        error = np.linalg.norm(PlanarArmEnv.forward_kinematics(q) - target)
        assert error < 1e-3
