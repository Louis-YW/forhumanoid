from __future__ import annotations

import numpy as np

from trace2touch.trajectory import Demonstration


def test_demonstration_round_trip(tmp_path) -> None:
    demo = Demonstration(
        time_s=np.linspace(0.0, 1.0, 8),
        object_xy_m=np.column_stack([np.linspace(0.0, 0.1, 8), np.linspace(0.1, 0.0, 8)]),
        confidence=np.linspace(0.5, 1.0, 8),
        workspace_size_m=(0.247, 0.160),
        source_sha256="a" * 64,
        metadata={"synthetic": True},
    )
    path = demo.save(tmp_path / "demo.npz")
    loaded = Demonstration.load(path)
    np.testing.assert_allclose(loaded.object_xy_m, demo.object_xy_m)
    assert loaded.metadata == demo.metadata
