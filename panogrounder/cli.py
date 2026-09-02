"""Command-line interface with intentionally fixed experiment presets."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .launcher import build_eval_command, build_train_command, shell_join
from .presets import EVAL_PRESETS, TRAIN_PRESETS
from .voting import format_report, vote_file


def _run_or_print(command: list[str], execute: bool) -> None:
    if execute:
        subprocess.run(command, check=True)
    else:
        print(shell_join(command))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="panogrounder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="build or run a canonical training job")
    train.add_argument("preset", choices=TRAIN_PRESETS)
    train.add_argument("--data", type=Path, action="append", required=True)
    train.add_argument("--model-path", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--hostfile", type=Path)
    train.add_argument("--execute", action="store_true")

    evaluate = subparsers.add_parser("eval", help="build or run a canonical evaluation job")
    evaluate.add_argument("preset", choices=EVAL_PRESETS)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--execute", action="store_true")

    vote = subparsers.add_parser("vote", help="aggregate view predictions")
    vote.add_argument("dataset", choices=("ScanRefer", "Sr3D", "Nr3D"))
    vote.add_argument("--input", type=Path, required=True)
    vote.add_argument("--output", type=Path, required=True)
    vote.add_argument("--metadata", type=Path)
    visibility = subparsers.add_parser("visibility", help="compute per-view visibility")
    visibility.add_argument("--annotations", type=Path, required=True)
    visibility.add_argument("--instance-dir", type=Path, required=True)
    visibility.add_argument("--render-mode", required=True)
    visibility.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        command = build_train_command(
            args.preset, args.data, args.model_path, args.output_dir,
            hostfile=args.hostfile,
        )
        _run_or_print(command, args.execute)
    elif args.command == "eval":
        command = build_eval_command(args.preset, args.data, args.checkpoint, args.output_dir)
        _run_or_print(command, args.execute)
    elif args.command == "vote":
        results = vote_file(args.input, args.output, args.dataset, args.metadata)
        report = format_report(results, args.dataset)
        args.output.with_suffix(".txt").write_text(report, encoding="utf-8")
        print(report, end="")
    else:
        from .visibility import compute_visibility_file

        compute_visibility_file(
            args.annotations, args.instance_dir, args.output, args.render_mode
        )


if __name__ == "__main__":
    main()
