"""
Run one or more evaluation episodes for the loop-only PPO model and save per-step wheel data.

Example:
  python test/eval_loop_wheels.py
"""

import argparse
import csv
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Main import (  # noqa: E402
    DEFAULT_MAX_EPISODE_STEPS,
    build_vec_env,
    close_vec_env_gracefully,
    map_engine_arg,
)
from dtps_shutdown_patch import apply_dtps_shutdown_patch  # noqa: E402
from start_stop_engine import start_engine, stop_engine  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runs_db21j_loop_only_V1/ppo_db21j_loop_only.zip",
        help="path to PPO model zip",
    )
    parser.add_argument("--map", type=str, default="loop", help="map name under --maps-dir")
    parser.add_argument("--maps-dir", type=str, default="./maps", help="local maps directory")
    parser.add_argument("--episodes", type=int, default=1, help="number of evaluation episodes")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="TimeLimit wrapper for one episode",
    )
    parser.add_argument("--entity-name", type=str, default="map_0/vehicle_0", help="target vehicle entity name")
    parser.add_argument("--device", type=str, default="auto", help="torch device: auto/cpu/cuda")
    parser.add_argument(
        "--respawn-mode",
        type=str,
        default="fixed",
        choices=("random", "fixed"),
        help="evaluation respawn mode",
    )
    parser.add_argument("--show-figure", action="store_true", help="show the local DB21J figure window")
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
        "--output-dir",
        type=str,
        default="test/loop_wheel_eval",
        help="directory for CSV and engine log output",
    )
    return parser.parse_args()


def _unwrap_vec_env(vec_env):
    env = vec_env
    seen = set()
    while hasattr(env, "venv") and id(env) not in seen:
        seen.add(id(env))
        env = env.venv
    if not hasattr(env, "envs") or not env.envs:
        raise RuntimeError("Failed to unwrap vectorized env to a concrete env")
    return env.envs[0]


def _unwrap_gym_env(env):
    cur = env
    seen = set()
    while True:
        cur_id = id(cur)
        if cur_id in seen:
            break
        seen.add(cur_id)
        if hasattr(cur, "env") and getattr(cur, "env") is not None:
            cur = cur.env
            continue
        try:
            unwrapped = cur.unwrapped
        except Exception:
            break
        if unwrapped is cur:
            break
        cur = unwrapped
    return cur


def _extract_pose(base_env):
    pose = getattr(base_env, "last_pose", None)
    if pose is None:
        return None, None, None
    pos = pose.get("position", {})
    return (
        float(pos.get("x", np.nan)),
        float(pos.get("y", np.nan)),
        float(pos.get("z", np.nan)),
    )


def main():
    args = parse_args()
    apply_dtps_shutdown_patch()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.map}_wheel_log.csv"
    engine_log_path = output_dir / f"{args.map}_engine.log"

    respawn_kwargs = {
        "lateral_jitter": 0.02,
        "yaw_jitter_deg": 0.0,
        "fallback_bbox": None,
        "avoid_junction": True,
        "max_spawn_angle_deg": 4.0,
    }
    reward_kwargs = {
        "reward_mode": "posangle",
        "include_velocity_reward": True,
    }

    engine_proc = None
    engine_log = None
    venv = None

    def cleanup(*_):
        nonlocal engine_proc, engine_log, venv
        if venv is not None:
            try:
                close_vec_env_gracefully(venv)
            except Exception:
                pass
            venv = None
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

    rows = []
    try:
        engine_log = open(engine_log_path, "w", encoding="utf-8")
        engine_proc, _, _ = start_engine(
            map_engine_arg(args.maps_dir, args.map),
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
                "DUCKIEMATRIX_RESPAWN_MODE": args.respawn_mode,
                "DUCKIEMATRIX_RESPAWN_MAX_SPAWN_ANGLE_DEG": str(respawn_kwargs["max_spawn_angle_deg"]),
            },
            stdout=engine_log,
            stderr=subprocess.STDOUT,
        )

        venv = build_vec_env(
            entity_name=args.entity_name,
            headless=not args.show_figure,
            max_episode_steps=args.max_episode_steps,
            respawn_mode=args.respawn_mode,
            respawn_kwargs=respawn_kwargs,
            reward_kwargs=reward_kwargs,
        )
        wrapped_env = _unwrap_vec_env(venv)
        base_env = _unwrap_gym_env(wrapped_env)
        model = PPO.load(str(model_path), env=venv, device=args.device)

        for episode_idx in range(1, args.episodes + 1):
            obs = venv.reset()
            done_flag = False
            episode_reward = 0.0
            step_idx = 0
            while not done_flag:
                action, _ = model.predict(obs, deterministic=True)
                heading = float(np.asarray(action).reshape(-1)[0])

                obs, reward, done, info = venv.step(action)
                reward_value = float(np.asarray(reward)[0])
                done_flag = bool(np.asarray(done)[0])
                info0 = info[0] if info else {}

                wheel_vels = np.asarray(getattr(base_env, "wheelVels", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
                pose_x, pose_y, pose_z = _extract_pose(base_env)
                custom_rewards = dict(info0.get("custom_rewards", {}))
                truncated = bool(info0.get("TimeLimit.truncated", False))
                terminated = bool(done_flag and not truncated)

                rows.append(
                    {
                        "episode": episode_idx,
                        "step": step_idx,
                        "heading_action": heading,
                        "wheel_left": float(wheel_vels[0]) if wheel_vels.size > 0 else np.nan,
                        "wheel_right": float(wheel_vels[1]) if wheel_vels.size > 1 else np.nan,
                        "reward": reward_value,
                        "orientation_reward": float(custom_rewards.get("orientation", np.nan)),
                        "velocity_reward": float(custom_rewards.get("velocity", np.nan)),
                        "dist_penalty": float(custom_rewards.get("dist_penalty", np.nan)),
                        "speed": float(info0.get("speed", np.nan)),
                        "lp_dist": float(info0.get("lp_dist", np.nan)) if info0.get("lp_dist") is not None else np.nan,
                        "lp_dot_dir": float(info0.get("lp_dot_dir", np.nan)) if info0.get("lp_dot_dir") is not None else np.nan,
                        "pose_x": pose_x,
                        "pose_y": pose_y,
                        "pose_z": pose_z,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                episode_reward += reward_value
                step_idx += 1

            print(f"[EP {episode_idx}/{args.episodes}] reward={episode_reward:.3f} steps={step_idx}")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

        print(f"[DONE] saved wheel log to: {csv_path}")
        print(f"[DONE] engine log: {engine_log_path}")
        if rows:
            print(
                "[DONE] "
                f"rows={len(rows)} "
                f"mean_reward={np.mean([r['reward'] for r in rows]):.4f} "
                f"mean_left={np.mean([r['wheel_left'] for r in rows]):.4f} "
                f"mean_right={np.mean([r['wheel_right'] for r in rows]):.4f}"
            )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
