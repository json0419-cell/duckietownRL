#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Main.py in loop-only mode.")
    parser.add_argument("--timesteps", type=int, default=300_000, help="total training timesteps")
    parser.add_argument("--segment-timesteps", type=int, default=16_384, help="segment size")
    parser.add_argument("--logdir", type=str, default="./runs_db21j_loop_only", help="output directory")
    parser.add_argument("--load-model", type=str, default=None, help="resume from an existing PPO .zip")
    parser.add_argument("--device", type=str, default="auto", help="torch device")
    parser.add_argument("--graphics-api", type=str, default="opengl", choices=("opengl", "vulkan", "default"))
    parser.add_argument("--show-figure", action="store_true", help="show DB21J matplotlib window")
    parser.add_argument("--pull", action="store_true", help="allow dts to pull images")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="extra arguments forwarded to Main.py; prefix with '--' if needed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    main_py = repo_root / "Main.py"

    cmd = [
        sys.executable,
        str(main_py),
        "--timesteps",
        str(args.timesteps),
        "--segment-timesteps",
        str(args.segment_timesteps),
        "--logdir",
        args.logdir,
        "--save-name",
        "ppo_db21j_loop_only",
        "--first-map",
        "loop",
        "--only-map",
        "loop",
        "--map-order",
        "round_robin",
        "--device",
        args.device,
        "--graphics-api",
        args.graphics_api,
    ]

    if args.load_model:
        cmd.extend(["--load-model", args.load_model])
    if args.show_figure:
        cmd.append("--show-figure")
    if args.pull:
        cmd.append("--pull")
    if args.extra_args:
        extras = args.extra_args
        if extras and extras[0] == "--":
            extras = extras[1:]
        cmd.extend(extras)

    print("[INFO] loop-only command:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
