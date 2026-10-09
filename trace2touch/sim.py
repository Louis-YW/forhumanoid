from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from .trajectory import RetargetedPlan


@dataclass(slots=True)
class RolloutResult:
    final_distance_m: float
    success: bool
    min_clearance_m: float
    control_energy: float
    puck_path: np.ndarray
    ee_path: np.ndarray
    joint_path: np.ndarray
    frames: list[np.ndarray]

    def summary(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("puck_path", "ee_path", "joint_path", "frames"):
            values.pop(key)
        return values


def _xml_for_plan(plan: RetargetedPlan, config: dict[str, Any]) -> str:
    sim = config["simulation"]
    dt = float(sim.get("dt", 0.005))
    puck_radius = float(sim.get("puck_radius_m", 0.038))
    ee_radius = float(sim.get("ee_radius_m", 0.035))
    puck_half_height = float(sim.get("puck_half_height_m", 0.018))
    obstacle_xml = []
    for index, obstacle in enumerate(plan.obstacles):
        obstacle_xml.append(
            f'<geom name="obstacle_{index}" type="cylinder" '
            f'pos="{obstacle["x"]} {obstacle["y"]} {obstacle["height"] / 2}" '
            f'size="{obstacle["radius"]} {obstacle["height"] / 2}" '
            'rgba="0.22 0.24 0.28 1" friction="1.2 0.01 0.001"/>'
        )
    obstacles = "\n".join(obstacle_xml)
    return f"""
<mujoco model="trace2touch_planar_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <global offwidth="720" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>
  <default>
    <joint type="hinge" axis="0 0 1" damping="4.0" armature="0.035" limited="true"/>
    <geom condim="4" friction="0.9 0.01 0.001" solref="0.015 1" solimp="0.90 0.98 0.002"/>
  </default>
  <asset>
    <texture name="floor" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.92 0.93 0.95" rgb2="0.78 0.80 0.84"/>
    <material name="floor" texture="floor" texrepeat="4 4" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
    <geom name="table" type="plane" size="1.1 0.85 0.1" material="floor" friction="1.0 0.01 0.001"/>
    <camera name="top" pos="0.0 0.0 1.55" xyaxes="1 0 0 0 1 0"/>
    <body name="arm_base" pos="-0.45 0 0.05">
      <geom type="cylinder" size="0.065 0.05" pos="0 0 -0.01" rgba="0.12 0.15 0.21 1"
            contype="0" conaffinity="0"/>
      <joint name="q1" range="-2.95 2.95"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0.32 0 0" size="0.025" rgba="0.18 0.42 0.72 1"/>
      <body name="joint2" pos="0.32 0 0">
        <joint name="q2" range="-2.75 2.75"/>
        <geom name="link2" type="capsule" fromto="0 0 0 0.28 0 0" size="0.023" rgba="0.24 0.50 0.80 1"/>
        <body name="joint3" pos="0.28 0 0">
          <joint name="q3" range="-2.75 2.75"/>
          <geom name="link3" type="capsule" fromto="0 0 0 0.22 0 0" size="0.021" rgba="0.31 0.59 0.86 1"/>
          <body name="tool" pos="0.22 0 0">
            <geom name="end_effector" type="sphere" size="{ee_radius}" rgba="0.08 0.10 0.14 1"/>
            <site name="ee_site" size="0.009" rgba="1 0.8 0.05 1"/>
          </body>
        </body>
      </body>
    </body>
    <body name="puck" pos="0 0 {puck_half_height}">
      <joint name="puck_x" type="slide" axis="1 0 0" limited="false" damping="0.20" armature="0.005"/>
      <joint name="puck_y" type="slide" axis="0 1 0" limited="false" damping="0.20" armature="0.005"/>
      <geom name="puck_geom" type="cylinder" size="{puck_radius} {puck_half_height}"
            mass="{float(sim.get("puck_mass_kg", 0.12))}" rgba="0.78 0.12 0.68 1"
            friction="{float(sim.get("puck_friction", 0.55))} 0.01 0.001"/>
    </body>
    <site name="goal" type="cylinder" pos="{plan.goal_xy[0]} {plan.goal_xy[1]} 0.002"
          size="{float(sim.get("success_threshold_m", 0.055))} 0.002" rgba="0.1 0.75 0.35 0.35"/>
    {obstacles}
  </worldbody>
  <actuator>
    <position name="a1" joint="q1" kp="90" ctrlrange="-2.95 2.95"/>
    <position name="a2" joint="q2" kp="80" ctrlrange="-2.75 2.75"/>
    <position name="a3" joint="q3" kp="60" ctrlrange="-2.75 2.75"/>
  </actuator>
</mujoco>
"""


class PlanarArmEnv:
    """Small MuJoCo environment used for fast CPU trajectory optimisation."""

    base_xy = np.array([-0.45, 0.0], dtype=float)
    link_lengths = np.array([0.32, 0.28, 0.22], dtype=float)

    def __init__(self, plan: RetargetedPlan, config: dict[str, Any]):
        self.plan = plan
        self.config = config
        self.model = mujoco.MjModel.from_xml_string(_xml_for_plan(plan, config))
        self.data = mujoco.MjData(self.model)
        self.joint_names = ("q1", "q2", "q3")
        self.qpos_indices = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in self.joint_names
            ]
        )
        puck_x_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_x")
        puck_y_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_y")
        self.puck_qpos_adrs = (
            int(self.model.jnt_qposadr[puck_x_joint_id]),
            int(self.model.jnt_qposadr[puck_y_joint_id]),
        )
        self.puck_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "puck")
        self.puck_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "puck_geom")
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._base_puck_mass = float(self.model.body_mass[self.puck_body_id])
        self._base_puck_friction = float(self.model.geom_friction[self.puck_geom_id, 0])
        self._base_actuator_gain = self.model.actuator_gainprm.copy()
        self._base_actuator_bias = self.model.actuator_biasprm.copy()
        self._base_dof_damping = self.model.dof_damping.copy()

    @classmethod
    def forward_kinematics(cls, q: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(np.asarray(q, dtype=float))
        return cls.base_xy + np.array(
            [
                np.sum(cls.link_lengths * np.cos(cumulative)),
                np.sum(cls.link_lengths * np.sin(cumulative)),
            ]
        )

    @classmethod
    def jacobian(cls, q: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(np.asarray(q, dtype=float))
        jacobian = np.zeros((2, 3), dtype=float)
        for column in range(3):
            jacobian[0, column] = -np.sum(cls.link_lengths[column:] * np.sin(cumulative[column:]))
            jacobian[1, column] = np.sum(cls.link_lengths[column:] * np.cos(cumulative[column:]))
        return jacobian

    @classmethod
    def inverse_kinematics(
        cls,
        target_xy: np.ndarray,
        initial_q: np.ndarray | None = None,
        *,
        iterations: int = 80,
        damping: float = 0.025,
    ) -> np.ndarray:
        target_xy = np.asarray(target_xy, dtype=float)
        relative = target_xy - cls.base_xy
        distance = float(np.linalg.norm(relative))
        # Analytic 2-link seed with links 2+3 aligned. It avoids the singular,
        # fully-extended seed that iterative IK otherwise falls into.
        l1 = float(cls.link_lengths[0])
        l23 = float(cls.link_lengths[1] + cls.link_lengths[2])
        cosine = np.clip((distance**2 - l1**2 - l23**2) / (2 * l1 * l23), -1.0, 1.0)
        seed_candidates = []
        for sign in (1.0, -1.0):
            q2 = sign * float(np.arccos(cosine))
            q1 = float(np.arctan2(relative[1], relative[0])) - float(
                np.arctan2(l23 * np.sin(q2), l1 + l23 * np.cos(q2))
            )
            seed_candidates.append(np.array([q1, q2, 0.0], dtype=float))
        if initial_q is None:
            q = min(seed_candidates, key=lambda candidate: np.linalg.norm(candidate))
        else:
            initial = np.asarray(initial_q, dtype=float)
            q = min(
                [initial, *seed_candidates],
                key=lambda candidate: (
                    np.linalg.norm(candidate - initial)
                    + 20.0 * np.linalg.norm(cls.forward_kinematics(candidate) - target_xy)
                ),
            ).copy()
        limits = np.array([2.94, 2.74, 2.74])
        q = np.clip(q, -limits, limits)
        for _ in range(iterations):
            error = target_xy - cls.forward_kinematics(q)
            if np.linalg.norm(error) < 2e-4:
                break
            jacobian = cls.jacobian(q)
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(2), error
            )
            # A small null-space bias keeps the redundant elbow away from joint limits.
            projector = np.eye(3) - jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(2), jacobian
            )
            update += projector @ (-0.015 * q)
            q = np.clip(q + np.clip(update, -0.18, 0.18), -limits, limits)
        if np.linalg.norm(target_xy - cls.forward_kinematics(q)) > 0.015:
            raise ValueError(f"target is outside the arm workspace: {target_xy}")
        return q

    def _reset(self, first_target: np.ndarray, seed: int, randomize: bool) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        if randomize:
            mass_scale = rng.uniform(0.75, 1.30)
            friction_scale = rng.uniform(0.70, 1.35)
            actuator_scale = rng.uniform(0.90, 1.10)
            damping_scale = rng.uniform(0.85, 1.15)
            start_noise = np.clip(rng.normal(0.0, 0.003, size=2), -0.006, 0.006)
        else:
            mass_scale = friction_scale = 1.0
            actuator_scale = damping_scale = 1.0
            start_noise = np.zeros(2)
        self.model.body_mass[self.puck_body_id] = self._base_puck_mass * mass_scale
        self.model.geom_friction[self.puck_geom_id, 0] = self._base_puck_friction * friction_scale
        self.model.actuator_gainprm[:] = self._base_actuator_gain
        self.model.actuator_biasprm[:] = self._base_actuator_bias
        self.model.actuator_gainprm[:, 0] *= actuator_scale
        self.model.actuator_biasprm[:, 1] *= actuator_scale
        self.model.dof_damping[:] = self._base_dof_damping
        self.model.dof_damping[:3] *= damping_scale
        initial_q = self.inverse_kinematics(first_target)
        self.data.qpos[self.qpos_indices] = initial_q
        self.data.qpos[self.puck_qpos_adrs[0]] = self.plan.start_xy[0] + start_noise[0]
        self.data.qpos[self.puck_qpos_adrs[1]] = self.plan.start_xy[1] + start_noise[1]
        self.data.ctrl[:] = initial_q
        mujoco.mj_forward(self.model, self.data)
        return initial_q

    def rollout(
        self,
        ee_targets: np.ndarray,
        *,
        seed: int = 0,
        randomize: bool = False,
        puck_targets: np.ndarray | None = None,
        feedback_gain: float = 0.0,
        render: bool = False,
        render_stride: int = 2,
    ) -> RolloutResult:
        ee_targets = np.asarray(ee_targets, dtype=float)
        previous_q = self._reset(ee_targets[0], seed, randomize)
        substeps = int(self.config["simulation"].get("substeps", 8))
        puck_path: list[np.ndarray] = []
        ee_path: list[np.ndarray] = []
        joint_path: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        renderer = None
        if render:
            renderer = mujoco.Renderer(self.model, height=640, width=640)

        energy = 0.0
        min_clearance = np.inf
        if puck_targets is not None and len(puck_targets) != len(ee_targets):
            raise ValueError("puck_targets and ee_targets must have equal length")
        max_feedback = float(self.config["simulation"].get("max_feedback_m", 0.04))
        try:
            for index, target in enumerate(ee_targets):
                controlled_target = target.copy()
                if puck_targets is not None and feedback_gain > 0:
                    puck_now = self.data.xpos[self.puck_body_id, :2]
                    # Shift the nominal contact pose with the observed puck, rather
                    # than chasing the reference ahead of it. If the puck lags, the
                    # pusher stays behind the puck instead of overtaking it.
                    correction = feedback_gain * (puck_now - puck_targets[index])
                    correction_norm = float(np.linalg.norm(correction))
                    if correction_norm > max_feedback:
                        correction *= max_feedback / correction_norm
                    controlled_target = controlled_target + correction
                q_target = self.inverse_kinematics(controlled_target, previous_q)
                waypoint_seconds = substeps * float(self.config["simulation"].get("dt", 0.005))
                max_step = (
                    float(self.config["simulation"].get("max_joint_speed_rad_s", 3.0))
                    * waypoint_seconds
                )
                q_target = previous_q + np.clip(q_target - previous_q, -max_step, max_step)
                control_start = self.data.ctrl.copy()
                for substep_index in range(substeps):
                    alpha = (substep_index + 1) / substeps
                    self.data.ctrl[:] = (1.0 - alpha) * control_start + alpha * q_target
                    before = self.data.qvel[self.qpos_indices].copy()
                    mujoco.mj_step(self.model, self.data)
                    energy += float(
                        np.sum((self.data.ctrl - self.data.qpos[self.qpos_indices]) ** 2)
                    )
                    energy += 1e-4 * float(np.sum(before**2))
                previous_q = q_target
                puck_xy = self.data.xpos[self.puck_body_id, :2].copy()
                ee_xy = self.data.site_xpos[self.ee_site_id, :2].copy()
                puck_path.append(puck_xy)
                ee_path.append(ee_xy)
                joint_path.append(self.data.qpos[self.qpos_indices].copy())
                for obstacle in self.plan.obstacles:
                    clearance = np.linalg.norm(ee_xy - np.array([obstacle["x"], obstacle["y"]])) - (
                        float(obstacle["radius"])
                        + float(self.config["simulation"].get("ee_radius_m", 0.035))
                    )
                    min_clearance = min(min_clearance, float(clearance))
                if renderer is not None and index % max(1, render_stride) == 0:
                    renderer.update_scene(self.data, camera="top")
                    frames.append(renderer.render().copy())
        finally:
            if renderer is not None:
                renderer.close()

        puck_array = np.asarray(puck_path)
        final_distance = float(np.linalg.norm(puck_array[-1] - self.plan.goal_xy))
        threshold = float(self.config["simulation"].get("success_threshold_m", 0.055))
        return RolloutResult(
            final_distance_m=final_distance,
            success=final_distance <= threshold,
            min_clearance_m=float(min_clearance if np.isfinite(min_clearance) else 1.0),
            control_energy=energy / max(1, len(ee_targets) * substeps),
            puck_path=puck_array,
            ee_path=np.asarray(ee_path),
            joint_path=np.asarray(joint_path),
            frames=frames,
        )
