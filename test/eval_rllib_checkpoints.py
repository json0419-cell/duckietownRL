"""
Batch-evaluate RLlib PPO checkpoints on one Duckiematrix map.

Example:
  python test/eval_rllib_checkpoints.py \
    --checkpoints-dir runs_db21j_multi_engine_rllib/checkpoints \
    --map _custom_technical_floor \
    --episodes 1 \
    --respawn-mode fixed
"""

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_rllib_model import build_env, resolve_policy_checkpoint
from core.env_builder import (
    DEFAULT_FORWARD_SPEED,
    DEFAULT_FRAME_REPEAT_PROB,
    DEFAULT_HEADING_TYPE,
    DEFAULT_LANE_MASK_NOISE_STRENGTH,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_OBSERVATION_MODE,
    DEFAULT_PHOTOMETRIC_AUG_STRENGTH,
    DEFAULT_YELLOW_LANE_AUG_STRENGTH,
    DEFAULT_MOTION_BLUR_KERNEL_SIZE,
    VALID_OBSERVATION_MODES,
)
from core.map_utils import discover_maps, map_engine_arg
from runtime.dtps_shutdown_patch import apply_dtps_shutdown_patch
from runtime.start_stop_engine import start_engine, stop_engine
from wrappers.action_wrappers import VALID_HEADING_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=str, required=True, help="directory containing RLlib checkpoint subdirs")
    parser.add_argument("--policy-id", type=str, default="default_policy", help="policy id inside the Algorithm checkpoint")
    parser.add_argument("--map", type=str, default="_custom_technical_floor", help="evaluation map name under --maps-dir")
    parser.add_argument("--maps-dir", type=str, default="./maps", help="local maps directory")
    parser.add_argument("--episodes", type=int, default=1, help="number of evaluation episodes per checkpoint")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="TimeLimit wrapper for one episode; use 0 to disable TimeLimit",
    )
    parser.add_argument("--entity-name", type=str, default="map_0/vehicle_0", help="target vehicle entity name")
    parser.add_argument("--frame-stack", type=int, default=3, help="number of RGB frames stacked along channels")
    parser.add_argument(
        "--forward-speed",
        type=float,
        default=DEFAULT_FORWARD_SPEED,
        help="global wheel-speed scale applied after heading-to-wheels mapping",
    )
    parser.add_argument(
        "--forward-speed-min",
        type=float,
        default=None,
        help="minimum episode-level sampled forward speed; requires --forward-speed-max",
    )
    parser.add_argument(
        "--forward-speed-max",
        type=float,
        default=None,
        help="maximum episode-level sampled forward speed; requires --forward-speed-min",
    )
    parser.add_argument(
        "--heading-type",
        type=str,
        default=DEFAULT_HEADING_TYPE,
        choices=VALID_HEADING_TYPES,
        help="scalar steering-to-wheel mapping used by the environment",
    )
    parser.add_argument(
        "--frame-repeat-prob",
        type=float,
        default=DEFAULT_FRAME_REPEAT_PROB,
        help="probability of repeating the previous resized observation frame",
    )
    parser.add_argument(
        "--motion-blur-kernel-size",
        type=int,
        default=DEFAULT_MOTION_BLUR_KERNEL_SIZE,
        help="Duckietown-RL-style rotational blur strength after resize; use 0 to disable",
    )
    parser.add_argument(
        "--photometric-aug-strength",
        type=float,
        default=DEFAULT_PHOTOMETRIC_AUG_STRENGTH,
        help="episode-level photometric augmentation strength after resize; use 0 to disable",
    )
    parser.add_argument(
        "--yellow-lane-aug-strength",
        type=float,
        default=DEFAULT_YELLOW_LANE_AUG_STRENGTH,
        help="episode-level targeted weakening of the yellow center line after resize; use 0 to disable",
    )
    parser.add_argument(
        "--observation-mode",
        type=str,
        default=DEFAULT_OBSERVATION_MODE,
        choices=VALID_OBSERVATION_MODES,
        help="observation representation used for evaluation",
    )
    parser.add_argument(
        "--lane-mask-noise-strength",
        type=float,
        default=DEFAULT_LANE_MASK_NOISE_STRENGTH,
        help="mask-noise strength after binary lane extraction; only used in binary_lane mode",
    )
    parser.add_argument("--show-figure", action="store_true", help="show the local matplotlib figure window from DB21JEnv")
    parser.add_argument(
        "--respawn-mode",
        type=str,
        default="fixed",
        choices=("random", "fixed"),
        help="evaluation respawn mode",
    )
    parser.add_argument(
        "--respawn-backend",
        type=str,
        default="engine",
        choices=("engine", "wrapper", "hybrid"),
        help="which side owns respawn selection/validation",
    )
    parser.add_argument("--engine-host", type=str, default="127.0.0.1", help="engine host")
    parser.add_argument("--engine-port", type=int, default=7501, help="engine DTPS port")
    parser.add_argument("--engine-ready-timeout", type=float, default=40.0, help="wait engine readiness timeout")
    parser.add_argument("--container-name", type=str, default="dts-matrix-engine", help="engine docker container name")
    parser.add_argument(
        "--graphics-api",
        type=str,
        default="opengl",
        choices=("opengl", "vulkan", "default"),
        help="renderer graphics API for standalone mode",
    )
    parser.add_argument(
        "--renderer-process-name",
        type=str,
        default="duckiematrix.x86_64",
        help="process name pattern used to stop the local renderer",
    )
    parser.add_argument("--pull", action="store_true", help="allow dts to pull instead of using --no-pull")
    parser.add_argument(
        "--engine-log",
        type=str,
        default=None,
        help="optional path for engine stdout/stderr log",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="optional JSON output path; default is <checkpoints-dir>/../eval_logs/checkpoint_eval_<map>_<respawn>.json",
    )
    return parser.parse_args()


def validate_args(args) -> tuple[Path, Path, Path, Path]:
    checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve()
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")

    maps_root = Path(args.maps_dir).expanduser().resolve()
    available_maps = discover_maps(maps_root)
    if args.map not in available_maps:
        raise ValueError(f"Map '{args.map}' not found under {maps_root}. Available maps: {available_maps}")

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_episode_steps < 0:
        raise ValueError("--max-episode-steps must be non-negative")

    if args.engine_log:
        engine_log_path = Path(args.engine_log).expanduser().resolve()
    else:
        engine_log_path = (checkpoints_dir.parent / "eval_logs" / f"checkpoint_eval_engine_{args.map}.log").resolve()
    engine_log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_json:
        summary_json = Path(args.summary_json).expanduser().resolve()
    else:
        summary_json = (
            checkpoints_dir.parent / "eval_logs" / f"checkpoint_eval_{args.map}_{args.respawn_mode}.json"
        ).resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir, maps_root, engine_log_path, summary_json


def checkpoint_sort_key(path: Path):
    name = path.name
    if name.endswith("_best"):
        return (1, float("inf"))
    if name.endswith("_final"):
        return (2, float("inf"))
    suffix = name.rsplit("_", 1)[-1]
    try:
        return (0, int(suffix))
    except ValueError:
        return (3, name)


def discover_checkpoints(checkpoints_dir: Path) -> list[Path]:
    items = [p for p in checkpoints_dir.iterdir() if p.is_dir() and (p / "rllib_checkpoint.json").is_file()]
    if not items:
        raise RuntimeError(f"No RLlib checkpoints found under {checkpoints_dir}")
    return sorted(items, key=checkpoint_sort_key)


def write_summary_json(summary_json: Path, summary: dict) -> None:
    tmp_path = summary_json.with_suffix(summary_json.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp_path.replace(summary_json)


def load_existing_results(summary_json: Path, summary: dict) -> dict[str, dict]:
    if not summary_json.is_file():
        return {}

    with summary_json.open("r", encoding="utf-8") as f:
        existing = json.load(f)

    required_keys = (
        "map",
        "respawn_mode",
        "respawn_backend",
        "forward_speed",
        "forward_speed_min",
        "forward_speed_max",
        "heading_type",
        "frame_repeat_prob",
        "motion_blur_kernel_size",
        "photometric_aug_strength",
        "yellow_lane_aug_strength",
        "observation_mode",
        "lane_mask_noise_strength",
        "episodes_per_checkpoint",
        "checkpoints_dir",
    )
    compatibility_defaults = {
        "forward_speed": float(DEFAULT_FORWARD_SPEED),
        "forward_speed_min": None,
        "forward_speed_max": None,
        "frame_repeat_prob": float(DEFAULT_FRAME_REPEAT_PROB),
        "motion_blur_kernel_size": int(DEFAULT_MOTION_BLUR_KERNEL_SIZE),
        "photometric_aug_strength": float(DEFAULT_PHOTOMETRIC_AUG_STRENGTH),
        "yellow_lane_aug_strength": float(DEFAULT_YELLOW_LANE_AUG_STRENGTH),
        "observation_mode": DEFAULT_OBSERVATION_MODE,
        "lane_mask_noise_strength": float(DEFAULT_LANE_MASK_NOISE_STRENGTH),
    }
    for key in required_keys:
        existing_value = existing.get(key, compatibility_defaults.get(key))
        if existing_value != summary.get(key):
            raise RuntimeError(
                f"Existing summary at {summary_json} does not match current run for key '{key}': "
                f"{existing_value!r} != {summary.get(key)!r}"
            )

    results_by_name = {}
    for item in existing.get("results", []):
        checkpoint_name = item.get("checkpoint")
        if checkpoint_name:
            results_by_name[checkpoint_name] = item
    return results_by_name


def evaluate_policy(policy, env, episodes: int) -> dict:
    episode_rewards = []
    episode_lengths = []
    terminated_count = 0
    time_limit_count = 0

    for _ in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0
        prev_action = None
        prev_reward = 0.0
        last_info = info or {}
        state = policy.get_initial_state()

        while not (terminated or truncated):
            action, state, _ = policy.compute_single_action(
                obs=obs,
                state=state,
                prev_action=prev_action,
                prev_reward=prev_reward,
                info=last_info,
                explore=False,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            reward_f = float(reward)
            episode_reward += reward_f
            episode_steps += 1
            prev_action = action
            prev_reward = reward_f
            last_info = info or {}

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        if truncated:
            time_limit_count += 1
        else:
            terminated_count += 1

    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_steps": float(np.mean(episode_lengths)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "terminated_count": int(terminated_count),
        "time_limit_count": int(time_limit_count),
        "episode_rewards": [float(v) for v in episode_rewards],
        "episode_lengths": [int(v) for v in episode_lengths],
    }


def main() -> int:
    args = parse_args()
    apply_dtps_shutdown_patch()
    checkpoints_dir, maps_root, engine_log_path, summary_json = validate_args(args)
    checkpoints = discover_checkpoints(checkpoints_dir)

    try:
        from ray.rllib.policy.policy import Policy
    except Exception as e:
        raise RuntimeError(
            "RLlib is not installed in the current environment. "
            "Install a Ray build that includes RLlib before running this script."
        ) from e

    engine_proc = None
    engine_log = None
    env = None

    def cleanup(*_):
        nonlocal engine_proc, engine_log, env
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            env = None
        if engine_proc is not None:
            stop_engine(
                engine_proc,
                args.container_name,
                stop_renderer=True,
                renderer_process_name=args.renderer_process_name,
            )
            engine_proc = None
        if engine_log is not None:
            try:
                engine_log.close()
            except Exception:
                pass
            engine_log = None

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    map_arg = map_engine_arg(str(maps_root), args.map)
    engine_respawn_mode = args.respawn_mode if args.respawn_backend != "wrapper" else "fixed"

    try:
        engine_log = open(engine_log_path, "w", encoding="utf-8")
        engine_proc, _, _ = start_engine(
            map_arg,
            host=args.engine_host,
            port=args.engine_port,
            entity_name=args.entity_name,
            ready_timeout=args.engine_ready_timeout,
            container_name=args.container_name,
            renderer_process_name=args.renderer_process_name,
            no_pull=not args.pull,
            engine_only=False,
            graphics_api=args.graphics_api,
            env_overrides={
                "DUCKIEMATRIX_RESPAWN_MODE": engine_respawn_mode,
                "DUCKIEMATRIX_RESPAWN_YAW_JITTER_DEG": "8.0",
                "DUCKIEMATRIX_RESPAWN_MAX_SPAWN_ANGLE_DEG": "8.0",
            },
            stdout=engine_log,
            stderr=subprocess.STDOUT,
        )

        env = build_env(args)

        summary = {
            "map": args.map,
            "respawn_mode": args.respawn_mode,
            "respawn_backend": args.respawn_backend,
            "forward_speed": float(args.forward_speed),
            "forward_speed_min": args.forward_speed_min,
            "forward_speed_max": args.forward_speed_max,
            "heading_type": args.heading_type,
            "frame_repeat_prob": float(args.frame_repeat_prob),
            "motion_blur_kernel_size": int(args.motion_blur_kernel_size),
            "photometric_aug_strength": float(args.photometric_aug_strength),
            "yellow_lane_aug_strength": float(args.yellow_lane_aug_strength),
            "observation_mode": args.observation_mode,
            "lane_mask_noise_strength": float(args.lane_mask_noise_strength),
            "episodes_per_checkpoint": int(args.episodes),
            "checkpoints_dir": str(checkpoints_dir),
            "engine_log": str(engine_log_path),
            "results": [],
        }
        results_by_name = load_existing_results(summary_json, summary)
        if results_by_name:
            print(f"[RESUME] loaded {len(results_by_name)} completed checkpoints from {summary_json}")
            summary["results"] = [results_by_name[p.name] for p in checkpoints if p.name in results_by_name]
            write_summary_json(summary_json, summary)

        print(f"[INFO] checkpoints_dir={checkpoints_dir}")
        print(f"[INFO] map={args.map} respawn_mode={args.respawn_mode} episodes={args.episodes}")
        print(f"[INFO] evaluating {len(checkpoints)} checkpoints")

        for idx, checkpoint_dir in enumerate(checkpoints, start=1):
            if checkpoint_dir.name in results_by_name:
                print(f"[SKIP {idx}/{len(checkpoints)}] {checkpoint_dir.name} (already completed)")
                continue
            policy_checkpoint = resolve_policy_checkpoint(checkpoint_dir, args.policy_id)
            print(f"[LOAD {idx}/{len(checkpoints)}] {checkpoint_dir.name}")
            policy = Policy.from_checkpoint(str(policy_checkpoint))
            metrics = evaluate_policy(policy, env, args.episodes)
            result = {
                "checkpoint": checkpoint_dir.name,
                "checkpoint_path": str(checkpoint_dir),
                "policy_checkpoint": str(policy_checkpoint),
                **metrics,
            }
            results_by_name[checkpoint_dir.name] = result
            summary["results"] = [results_by_name[p.name] for p in checkpoints if p.name in results_by_name]
            write_summary_json(summary_json, summary)
            print(
                f"[RESULT {idx}/{len(checkpoints)}] "
                f"{checkpoint_dir.name} "
                f"mean_reward={metrics['mean_reward']:.3f} "
                f"std_reward={metrics['std_reward']:.3f} "
                f"mean_steps={metrics['mean_steps']:.1f} "
                f"time_limit={metrics['time_limit_count']} "
                f"terminated={metrics['terminated_count']}"
            )
        write_summary_json(summary_json, summary)
        print(f"[DONE] summary_json={summary_json}")
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
