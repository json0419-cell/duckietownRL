#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "./runs_db21j_loop_only_V1/ppo_db21j_seg19_loop.zip"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune loop-only training starting from seg19.")
    parser.add_argument("--timesteps", type=int, default=150_000, help="total fine-tuning timesteps")
    parser.add_argument("--segment-timesteps", type=int, default=16_384, help="segment size")
    parser.add_argument(
        "--logdir",
        type=str,
        default="./runs_db21j_loop_only_target35",
        help="output directory",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=DEFAULT_MODEL,
        help="starting PPO checkpoint",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="fine-tuning PPO learning rate")
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
    load_model = Path(args.load_model).expanduser().resolve()
    if not load_model.is_file():
        raise FileNotFoundError(f"Model file not found: {load_model}")

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
        "--learning-rate",
        str(args.learning_rate),
        "--graphics-api",
        args.graphics_api,
        "--load-model",
        str(load_model),
    ]

    if args.show_figure:
        cmd.append("--show-figure")
    if args.pull:
        cmd.append("--pull")
    if args.extra_args:
        extras = args.extra_args
        if extras and extras[0] == "--":
            extras = extras[1:]
        cmd.extend(extras)

    print("[INFO] loop seg19 fine-tune command:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
