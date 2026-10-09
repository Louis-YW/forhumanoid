from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .sim import PlanarArmEnv, RolloutResult
from .trajectory import RetargetedPlan


@dataclass(slots=True)
class OptimisationResult:
    path: np.ndarray
    residual: np.ndarray
    best_cost: float
    history: list[dict[str, float]]
    rollout: RolloutResult


def make_straight_baseline(plan: RetargetedPlan, config: dict[str, Any]) -> np.ndarray:
    """Construct the obvious direct-push baseline using no trajectory shape information."""
    n = len(plan.ee_xy)
    approach_n = int(config["retarget"].get("approach_samples", 10))
    follow_n = int(config["retarget"].get("follow_through_samples", 8))
    approach_n = min(max(0, approach_n), n - 2)
    follow_n = min(max(0, follow_n), n - approach_n - 2)
    direction = plan.goal_xy - plan.start_xy
    direction /= max(np.linalg.norm(direction), 1e-8)
    contact_distance = float(plan.transform["contact_distance_m"])
    contact_start = plan.start_xy - contact_distance * direction
    contact_goal = plan.goal_xy - contact_distance * direction
    push_n = n - approach_n - follow_n
    parts = []
    if approach_n:
        approach = np.linspace(plan.ee_xy[0], contact_start, approach_n, endpoint=False)
        parts.append(approach)
    parts.append(np.linspace(contact_start, contact_goal, push_n))
    if follow_n:
        distance = float(config["retarget"].get("follow_through_distance_m", 0.065))
        parts.append(
            contact_goal[None, :]
            + np.linspace(0.0, distance, follow_n + 1, endpoint=True)[1:, None] * direction
        )
    return np.vstack(parts)


def make_straight_object_targets(plan: RetargetedPlan, config: dict[str, Any]) -> np.ndarray:
    """Desired puck states paired with the no-demonstration straight baseline."""
    n = len(plan.ee_xy)
    approach_n = min(int(config["retarget"].get("approach_samples", 10)), n - 2)
    follow_n = min(
        int(config["retarget"].get("follow_through_samples", 8)),
        n - approach_n - 2,
    )
    push_n = n - approach_n - follow_n
    return np.vstack(
        [
            np.repeat(plan.start_xy[None, :], approach_n, axis=0),
            np.linspace(plan.start_xy, plan.goal_xy, push_n),
            np.repeat(plan.goal_xy[None, :], follow_n, axis=0),
        ]
    )


def _residual_curve(parameters: np.ndarray, n: int, knots: int) -> np.ndarray:
    internal = parameters.reshape(knots, 2)
    knot_t = np.linspace(0.0, 1.0, knots + 2)
    values = np.vstack([np.zeros((1, 2)), internal, np.zeros((1, 2))])
    query_t = np.linspace(0.0, 1.0, n)
    return CubicSpline(knot_t, values, axis=0, bc_type="natural")(query_t)


def _trajectory_roughness(path: np.ndarray) -> float:
    if len(path) < 4:
        return 0.0
    acceleration = np.diff(path, n=2, axis=0)
    jerk = np.diff(path, n=3, axis=0)
    return float(np.mean(np.sum(acceleration**2, axis=1)) + 0.5 * np.mean(np.sum(jerk**2, axis=1)))


def optimise_residual(
    plan: RetargetedPlan,
    config: dict[str, Any],
    *,
    progress: Callable[[dict[str, float]], None] | None = None,
) -> OptimisationResult:
    """CEM searches small corrections around the applicant-derived trajectory prior."""
    options = config["optimization"]
    knots = int(options.get("knots", 5))
    population = int(options.get("population", 48))
    elite_fraction = float(options.get("elite_fraction", 0.18))
    elite_count = max(2, round(population * elite_fraction))
    iterations = int(options.get("iterations", 6))
    rng = np.random.default_rng(int(options.get("seed", 11)))
    mean = np.zeros(knots * 2, dtype=float)
    std = np.full(knots * 2, float(options.get("initial_std_m", 0.035)), dtype=float)
    max_residual = float(options.get("max_residual_m", 0.10))
    env = PlanarArmEnv(plan, config)
    best_cost = np.inf
    best_path = plan.ee_xy.copy()
    best_residual = np.zeros_like(plan.ee_xy)
    feedback_gain = float(config["simulation"].get("feedback_gain", 0.0))
    best_rollout = env.rollout(plan.ee_xy, puck_targets=plan.object_xy, feedback_gain=feedback_gain)
    history: list[dict[str, float]] = []

    final_weight = float(options.get("final_distance_weight", 16.0))
    collision_weight = float(options.get("collision_weight", 30.0))
    clearance_margin = float(options.get("clearance_margin_m", 0.0))
    prior_weight = float(options.get("prior_weight", 0.8))
    roughness_weight = float(options.get("roughness_weight", 20.0))
    energy_weight = float(options.get("energy_weight", 0.015))
    domain_rollouts = max(1, int(options.get("domain_randomization_rollouts", 1)))

    for iteration in range(iterations):
        samples = rng.normal(mean, std, size=(population, mean.size))
        samples[0] = mean
        costs = np.full(population, np.inf, dtype=float)
        for sample_index, sample in enumerate(samples):
            residual = np.clip(
                _residual_curve(sample, len(plan.ee_xy), knots),
                -max_residual,
                max_residual,
            )
            candidate = plan.ee_xy + residual
            try:
                rollouts = [
                    env.rollout(
                        candidate,
                        seed=0,
                        randomize=False,
                        puck_targets=plan.object_xy,
                        feedback_gain=feedback_gain,
                    )
                ]
                rollouts.extend(
                    env.rollout(
                        candidate,
                        seed=1000 + robust_index,
                        randomize=True,
                        puck_targets=plan.object_xy,
                        feedback_gain=feedback_gain,
                    )
                    for robust_index in range(1, domain_rollouts)
                )
            except ValueError:
                continue
            rollout = rollouts[0]
            distances = np.asarray([item.final_distance_m for item in rollouts], dtype=float)
            mean_distance = float(distances.mean())
            tail_weight = float(options.get("robust_tail_weight", 0.0))
            final_distance = mean_distance + tail_weight * float(distances.max() - mean_distance)
            clearance_shortfalls = np.asarray(
                [max(0.0, clearance_margin - item.min_clearance_m) for item in rollouts],
                dtype=float,
            )
            mean_shortfall = float(clearance_shortfalls.mean())
            clearance_cost = mean_shortfall + tail_weight * float(
                clearance_shortfalls.max() - mean_shortfall
            )
            control_energy = float(np.mean([item.control_energy for item in rollouts]))
            confidence_weight = 0.2 + 0.8 * plan.confidence
            prior_cost = float(np.mean(confidence_weight * np.sum(residual**2, axis=1)))
            cost = (
                final_weight * final_distance
                + collision_weight * clearance_cost**2
                + prior_weight * prior_cost
                + roughness_weight * _trajectory_roughness(candidate)
                + energy_weight * control_energy
            )
            costs[sample_index] = cost
            if cost < best_cost:
                best_cost = float(cost)
                best_path, best_residual, best_rollout = candidate, residual, rollout

        finite = np.flatnonzero(np.isfinite(costs))
        if len(finite) < elite_count:
            raise RuntimeError("CEM could not find enough reachable trajectories")
        elite_indices = finite[np.argsort(costs[finite])[:elite_count]]
        elite = samples[elite_indices]
        smoothing = float(options.get("distribution_smoothing", 0.25))
        mean = smoothing * mean + (1.0 - smoothing) * elite.mean(axis=0)
        std = smoothing * std + (1.0 - smoothing) * (elite.std(axis=0) + 1e-4)
        record = {
            "iteration": float(iteration),
            "best_cost": float(best_cost),
            "elite_mean_cost": float(costs[elite_indices].mean()),
            "std_mean_m": float(std.mean()),
        }
        history.append(record)
        if progress is not None:
            progress(record)

    return OptimisationResult(
        path=best_path,
        residual=best_residual,
        best_cost=best_cost,
        history=history,
        rollout=best_rollout,
    )
