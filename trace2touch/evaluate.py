from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .optimize import (
    OptimisationResult,
    make_straight_baseline,
    make_straight_object_targets,
    optimise_residual,
)
from .retarget import retarget_demonstration
from .sim import PlanarArmEnv, RolloutResult
from .trajectory import Demonstration, RetargetedPlan


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "final_distance_mean_m": float(np.mean([row["final_distance_m"] for row in rows])),
        "final_distance_std_m": float(np.std([row["final_distance_m"] for row in rows])),
        "min_clearance_mean_m": float(np.mean([row["min_clearance_m"] for row in rows])),
        "control_energy_mean": float(np.mean([row["control_energy"] for row in rows])),
    }


def _plot_paths(
    plan: RetargetedPlan,
    rollouts: dict[str, RolloutResult],
    policies: dict[str, np.ndarray],
    output: Path,
) -> None:
    colours = {"straight": "#8d99ae", "video_prior": "#7b2cbf", "optimized": "#e63946"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(plan.object_xy[:, 0], plan.object_xy[:, 1], "k--", lw=2, label="recorded object path")
    for name, path in policies.items():
        ax.plot(path[:, 0], path[:, 1], color=colours[name], lw=2, label=name.replace("_", " "))
    for obstacle in plan.obstacles:
        circle = plt.Circle(
            (obstacle["x"], obstacle["y"]), obstacle["radius"], color="#343a40", alpha=0.65
        )
        ax.add_patch(circle)
    ax.scatter(*plan.start_xy, s=80, c="#277da1", marker="o", label="puck start", zorder=5)
    ax.scatter(*plan.goal_xy, s=120, c="#43aa8b", marker="*", label="goal", zorder=5)
    ax.set_title("Commanded end-effector paths")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    for name, rollout in rollouts.items():
        ax.plot(
            rollout.puck_path[:, 0],
            rollout.puck_path[:, 1],
            color=colours[name],
            lw=2.5,
            label=f"{name.replace('_', ' ')} ({rollout.final_distance_m * 100:.1f} cm)",
        )
    for obstacle in plan.obstacles:
        circle = plt.Circle(
            (obstacle["x"], obstacle["y"]), obstacle["radius"], color="#343a40", alpha=0.65
        )
        ax.add_patch(circle)
    ax.scatter(*plan.start_xy, s=80, c="#277da1", marker="o", zorder=5)
    ax.scatter(*plan.goal_xy, s=120, c="#43aa8b", marker="*", zorder=5)
    ax.set_title("Executed puck trajectories (nominal physics)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("Personal video as a behaviour prior; simulation only changes the residual")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_optimisation(result: OptimisationResult, output: Path) -> None:
    iterations = [int(row["iteration"]) + 1 for row in result.history]
    best = [row["best_cost"] for row in result.history]
    elite = [row["elite_mean_cost"] for row in result.history]
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(iterations, elite, "o--", label="elite mean")
    ax.plot(iterations, best, "o-", label="best seen")
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("objective (lower is better)")
    ax.set_title("Residual policy search")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_report(
    output: Path,
    demo: Demonstration,
    aggregate: dict[str, dict[str, float]],
    optimisation: OptimisationResult,
    config: dict[str, Any],
) -> None:
    provenance_label = (
        "synthetic smoke test" if demo.metadata.get("synthetic") else "personal recording"
    )
    lines = [
        "# Experiment report",
        "",
        f"Data source: **{provenance_label}**",
        f"SHA-256: `{demo.source_sha256}`",
        f"Valid colour detections: {100 * float(demo.metadata.get('valid_detection_fraction', 0)):.1f}%",
        f"Per-frame marker calibration: {100 * float(demo.metadata.get('per_frame_calibration_fraction', 0)):.1f}%",
        f"Evaluation domains: {int(config['evaluation'].get('seeds', 12))}",
        "",
        "| Controller | Success rate | Final error (mean ± std) | Mean pusher clearance |",
        "|---|---:|---:|---:|",
    ]
    for name in ("straight", "video_prior", "optimized"):
        row = aggregate[name]
        lines.append(
            f"| {name.replace('_', ' ').title()} | {100 * row['success_rate']:.0f}% | "
            f"{100 * row['final_distance_mean_m']:.1f} ± {100 * row['final_distance_std_m']:.1f} cm | "
            f"{100 * row['min_clearance_mean_m']:.1f} cm |"
        )
    lines.extend(
        [
            "",
            f"CEM best objective: `{optimisation.best_cost:.4f}`.",
            "",
            (
                "The randomized evaluation changes puck mass, friction, start pose, actuator "
                "gain, and joint damping for every seed. The video-prior controller retains "
                "the recorded path; the optimized controller may only add a smooth, bounded "
                "residual."
            ),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    demo: Demonstration,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    render: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = retarget_demonstration(demo, config)
    plan.save(output_dir / "retargeted_plan.npz")
    straight = make_straight_baseline(plan, config)
    optimisation = optimise_residual(
        plan,
        config,
        progress=lambda row: print(
            f"CEM {int(row['iteration']) + 1}: best={row['best_cost']:.4f}, "
            f"elite={row['elite_mean_cost']:.4f}"
        ),
    )
    policies = {
        "straight": straight,
        "video_prior": plan.ee_xy,
        "optimized": optimisation.path,
    }
    object_targets = {
        "straight": make_straight_object_targets(plan, config),
        "video_prior": plan.object_xy,
        "optimized": plan.object_xy,
    }
    np.savez_compressed(output_dir / "policies.npz", **policies)

    seeds = list(range(int(config["evaluation"].get("seeds", 12))))
    env = PlanarArmEnv(plan, config)
    rows: list[dict[str, Any]] = []
    nominal: dict[str, RolloutResult] = {}
    feedback_gain = float(config["simulation"].get("feedback_gain", 0.0))
    for name, path in policies.items():
        nominal[name] = env.rollout(
            path,
            seed=0,
            randomize=False,
            puck_targets=object_targets[name],
            feedback_gain=feedback_gain,
        )
        for seed in seeds:
            result = env.rollout(
                path,
                seed=seed,
                randomize=True,
                puck_targets=object_targets[name],
                feedback_gain=feedback_gain,
            )
            rows.append({"controller": name, "seed": seed, **result.summary()})

    aggregate = {
        name: _aggregate([row for row in rows if row["controller"] == name]) for name in policies
    }
    payload = {
        "provenance": {
            "source_sha256": demo.source_sha256,
            "synthetic": bool(demo.metadata.get("synthetic", False)),
            "collected_by": demo.metadata.get("collected_by", "unspecified"),
            "collection_date": demo.metadata.get("collection_date", "unspecified"),
            "source_filename": demo.metadata.get("source_filename", "unspecified"),
            "valid_detection_fraction": demo.metadata.get("valid_detection_fraction"),
            "per_frame_calibration_fraction": demo.metadata.get("per_frame_calibration_fraction"),
        },
        "evaluation": {
            "seeds": seeds,
            "success_threshold_m": float(config["simulation"].get("success_threshold_m", 0.055)),
            "clearance_metric": "minimum pusher-to-obstacle surface distance",
        },
        "aggregate": aggregate,
        "cem": {"best_cost": optimisation.best_cost, "history": optimisation.history},
        "nominal": {name: result.summary() for name, result in nominal.items()},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _plot_paths(plan, nominal, policies, output_dir / "trajectory_comparison.png")
    _plot_optimisation(optimisation, output_dir / "cem_convergence.png")
    _write_report(output_dir / "REPORT.md", demo, aggregate, optimisation, config)

    if render:
        rendered = env.rollout(
            optimisation.path,
            render=True,
            puck_targets=plan.object_xy,
            feedback_gain=feedback_gain,
        )
        if rendered.frames:
            imageio.mimsave(
                output_dir / "optimized_rollout.gif",
                rendered.frames,
                fps=float(config["evaluation"].get("gif_fps", 12)),
                loop=0,
            )
    return payload
