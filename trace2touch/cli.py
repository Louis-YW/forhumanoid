from __future__ import annotations

import argparse
import json
from pathlib import Path

from .board import make_capture_board
from .config import load_config
from .evaluate import run_experiment
from .synthetic import make_synthetic_video
from .trajectory import Demonstration
from .vision import extract_demonstration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace2touch",
        description="Retarget one personally recorded phone video to a MuJoCo robot arm.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    board = subparsers.add_parser("make-board", help="generate the printable A4 capture board")
    board.add_argument("--output", default="artifacts/capture_board.pdf")

    synthetic = subparsers.add_parser("make-synthetic", help="generate a software smoke-test video")
    synthetic.add_argument("--config", default="configs/synthetic.yaml")
    synthetic.add_argument("--output", default="data/raw/synthetic_demo.mp4")

    extract = subparsers.add_parser("extract", help="extract a calibrated trajectory from a video")
    extract.add_argument("--config", required=True)
    extract.add_argument("--video", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--overlay", default=None)

    run = subparsers.add_parser("run", help="retarget, optimise, evaluate and render")
    run.add_argument("--config", required=True)
    run.add_argument("--demo", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--no-render", action="store_true")

    pipeline = subparsers.add_parser("pipeline", help="run extraction and the full experiment")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--video", required=True)
    pipeline.add_argument("--output-dir", required=True)
    pipeline.add_argument("--no-render", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "make-board":
        path = make_capture_board(args.output)
        print(path.resolve())
        return
    if args.command == "make-synthetic":
        config = load_config(args.config)
        path = make_synthetic_video(args.output, config)
        print(path.resolve())
        return

    config = load_config(args.config)
    if args.command == "extract":
        demo = extract_demonstration(args.video, config, overlay_path=args.overlay)
        demo.save(args.output)
        print(json.dumps(demo.metadata, indent=2, default=str))
        return
    if args.command == "run":
        demo = Demonstration.load(args.demo)
        metrics = run_experiment(demo, config, args.output_dir, render=not args.no_render)
        print(json.dumps(metrics["aggregate"], indent=2))
        return
    if args.command == "pipeline":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        demo = extract_demonstration(
            args.video, config, overlay_path=output_dir / "tracking_overlay.gif"
        )
        demo.save(output_dir / "demonstration.npz")
        metrics = run_experiment(demo, config, output_dir, render=not args.no_render)
        print(json.dumps(metrics["aggregate"], indent=2))


if __name__ == "__main__":
    main()
