"""
Run a trained PPO model in Duckiematrix with the same wrappers used in Main.py.

Example:
  python test/eval_model.py --model runs_db21j_ppo_old/ppo_db21j_final.zip --map loop
"""

import argparse
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Main import DEFAULT_MAX_EPISODE_STEPS, build_vec_env, discover_maps, map_engine_arg
from start_stop_engine import start_engine, stop_engine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to PPO model zip")
    parser.add_argument("--map", type=str, default="loop", help="evaluation map name under --maps-dir")
    parser.add_argument("--maps-dir", type=str, default="./maps", help="local maps directory")
    parser.add_argument("--episodes", type=int, default=3, help="number of evaluation episodes")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="TimeLimit wrapper for one episode; use 0 to disable TimeLimit",
    )
    parser.add_argument("--entity-name", type=str, default="map_0/vehicle_0", help="target vehicle entity name")
    parser.add_argument(
        "--show-figure",
        action="store_true",
        help="show the local matplotlib figure window from DB21JEnv",
    )
    parser.add_argument("--device", type=str, default="auto", help="torch device: auto/cpu/cuda")
    parser.add_argument(
        "--respawn-mode",
        type=str,
        default="fixed",
        choices=("random", "fixed"),
        help="evaluation respawn mode; fixed is easier for quick visual inspection",
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
        help="optional path for engine stdout/stderr log; default is runs_db21j_ppo_old/engine_logs/eval_<map>.log",
    )
    return parser.parse_args()


def validate_args(args) -> tuple[Path, Path, Path]:
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

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
        engine_log_path = (Path("./runs_db21j_ppo_old") / "engine_logs" / f"eval_{args.map}.log").resolve()
    engine_log_path.parent.mkdir(parents=True, exist_ok=True)
    return model_path, maps_root, engine_log_path


def main():
    args = parse_args()
    model_path, maps_root, engine_log_path = validate_args(args)

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
                venv.close()
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

    map_arg = map_engine_arg(str(maps_root), args.map)

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
            engine_host=args.engine_host,
            engine_port=args.engine_port,
        )
        model = PPO.load(str(model_path), env=venv, device=args.device)

        print(f"[INFO] model={model_path}")
        print(f"[INFO] map={args.map}")
        print(f"[INFO] engine_log={engine_log_path}")

        episode_rewards = []
        episode_lengths = []
        for episode_idx in range(1, args.episodes + 1):
            obs = venv.reset()
            done_flag = False
            episode_reward = 0.0
            episode_steps = 0
            last_info = {}

            while not done_flag:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = venv.step(action)
                episode_reward += float(np.asarray(reward)[0])
                done_flag = bool(np.asarray(done)[0])
                episode_steps += 1
                if info and info[0]:
                    last_info = info[0]

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)

            terminated_reason = "unknown"
            if "TimeLimit.truncated" in last_info and last_info["TimeLimit.truncated"]:
                terminated_reason = "time_limit"
            elif last_info.get("episode"):
                terminated_reason = "terminated"

            print(
                f"[EP {episode_idx}/{args.episodes}] "
                f"reward={episode_reward:.3f} steps={episode_steps} end={terminated_reason}"
            )

        print(
            "[DONE] "
            f"mean_reward={np.mean(episode_rewards):.3f} "
            f"std_reward={np.std(episode_rewards):.3f} "
            f"mean_steps={np.mean(episode_lengths):.1f}"
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
