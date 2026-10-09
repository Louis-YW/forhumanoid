from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class Demonstration:
    """A calibrated object trajectory extracted from an applicant's video."""

    time_s: np.ndarray
    object_xy_m: np.ndarray
    confidence: np.ndarray
    workspace_size_m: tuple[float, float]
    source_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n = len(self.time_s)
        if n < 4:
            raise ValueError("a demonstration needs at least four samples")
        if self.object_xy_m.shape != (n, 2):
            raise ValueError("object_xy_m must have shape [N, 2]")
        if self.confidence.shape != (n,):
            raise ValueError("confidence must have shape [N]")
        if not np.all(np.isfinite(self.object_xy_m)):
            raise ValueError("trajectory contains non-finite coordinates")
        if np.any(np.diff(self.time_s) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("confidence must be in [0, 1]")

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            time_s=self.time_s,
            object_xy_m=self.object_xy_m,
            confidence=self.confidence,
            workspace_size_m=np.asarray(self.workspace_size_m),
            source_sha256=np.asarray(self.source_sha256),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True, default=str)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Demonstration:
        with np.load(path, allow_pickle=False) as data:
            demo = cls(
                time_s=data["time_s"].astype(float),
                object_xy_m=data["object_xy_m"].astype(float),
                confidence=data["confidence"].astype(float),
                workspace_size_m=tuple(data["workspace_size_m"].astype(float)),
                source_sha256=str(data["source_sha256"]),
                metadata=json.loads(str(data["metadata_json"])),
            )
        demo.validate()
        return demo


@dataclass(slots=True)
class RetargetedPlan:
    """Video-derived object and end-effector paths expressed in simulation metres."""

    object_xy: np.ndarray
    ee_xy: np.ndarray
    confidence: np.ndarray
    start_xy: np.ndarray
    goal_xy: np.ndarray
    obstacles: list[dict[str, float]]
    transform: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            object_xy=self.object_xy,
            ee_xy=self.ee_xy,
            confidence=self.confidence,
            start_xy=self.start_xy,
            goal_xy=self.goal_xy,
            obstacles_json=np.asarray(json.dumps(self.obstacles, sort_keys=True)),
            transform_json=np.asarray(json.dumps(self.transform, sort_keys=True)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> RetargetedPlan:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                object_xy=data["object_xy"].astype(float),
                ee_xy=data["ee_xy"].astype(float),
                confidence=data["confidence"].astype(float),
                start_xy=data["start_xy"].astype(float),
                goal_xy=data["goal_xy"].astype(float),
                obstacles=json.loads(str(data["obstacles_json"])),
                transform=json.loads(str(data["transform_json"])),
            )
