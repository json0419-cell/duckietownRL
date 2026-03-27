#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "./runs_db21j_generalize_loop_curling/ppo_db21j_seg06_curling.zip"
DEFAULT_STAGES = [
    {
        "name": "stage01_loop_curling",
        "maps": ["loop", "curling"],
        "timesteps": 150_000,
        "segment_timesteps": 8_192,
        "mode": "map_subset",
        "first_map": "loop",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the loop+curling stage from a selected checkpoint.")
    parser.add_argument(
        "--load-model",
        type=str,
        default=DEFAULT_MODEL,
        help="starting PPO checkpoint",
    )
    parser.add_argument(
        "--timesteps-per-stage",
        type=int,
        default=None,
        help="override training timesteps for the loop+curling stage",
    )
    parser.add_argument(
        "--segment-timesteps",
        type=int,
        default=None,
        help="override segment size for the loop+curling stage",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./runs_db21j_generalize_loop_curling_ft",
        help="output directory",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="PPO learning rate")
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
    stage_dir = Path(args.logdir).expanduser().resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)

    current_model = Path(args.load_model).expanduser().resolve()
    if not current_model.is_file():
        raise FileNotFoundError(f"Model file not found: {current_model}")

    extras = list(args.extra_args)
    if extras and extras[0] == "--":
        extras = extras[1:]

    stage = DEFAULT_STAGES[0]
    save_name = "ppo_db21j_loop_curling"
    stage_timesteps = (
        int(args.timesteps_per_stage)
        if args.timesteps_per_stage is not None
        else int(stage["timesteps"])
    )
    segment_timesteps = (
        int(args.segment_timesteps)
        if args.segment_timesteps is not None
        else int(stage["segment_timesteps"])
    )
    cmd = [
        sys.executable,
        str(main_py),
        "--timesteps",
        str(stage_timesteps),
        "--segment-timesteps",
        str(segment_timesteps),
        "--logdir",
        str(stage_dir),
        "--save-name",
        save_name,
        "--first-map",
        stage["first_map"],
        "--map-order",
        "round_robin",
        "--device",
        args.device,
        "--learning-rate",
        str(args.learning_rate),
        "--graphics-api",
        args.graphics_api,
        "--load-model",
        str(current_model),
        "--map-subset",
        ",".join(stage["maps"]),
    ]

    if args.show_figure:
        cmd.append("--show-figure")
    if args.pull:
        cmd.append("--pull")
    if extras:
        cmd.extend(extras)

    print(
        f"[INFO] loop+curling stage: maps={stage['maps']} timesteps={stage_timesteps} "
        f"segment_timesteps={segment_timesteps}"
    )
    print("[INFO] command:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(repo_root))
    if rc != 0:
        return rc

    current_model = stage_dir / f"{save_name}.zip"
    if not current_model.is_file():
        raise FileNotFoundError(f"Expected output model not found: {current_model}")

    print(f"[DONE] final model: {current_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
