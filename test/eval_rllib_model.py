"""
Evaluate an RLlib PPO checkpoint on a single Duckiematrix map.

Example:
  python test/eval_rllib_model.py \
    --checkpoint runs_db21j_multi_engine_rllib/checkpoints/rllib_db21j_multi_engine_best \
    --map _custom_technical_floor
"""

import argparse
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.env_builder import (
    DEFAULT_FORWARD_SPEED,
    DEFAULT_FRAME_REPEAT_PROB,
    DEFAULT_HEADING_TYPE,
    DEFAULT_LANE_MASK_NOISE_STRENGTH,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_MAX_STEER,
    DEFAULT_OBSERVATION_MODE,
    DEFAULT_PHOTOMETRIC_AUG_STRENGTH,
    DEFAULT_YELLOW_LANE_AUG_STRENGTH,
    DEFAULT_MOTION_BLUR_KERNEL_SIZE,
    VALID_OBSERVATION_MODES,
    make_single_env,
)
from core.frame_stack import ChannelFrameStack
from core.map_utils import discover_maps, map_engine_arg
from runtime.dtps_shutdown_patch import apply_dtps_shutdown_patch
from runtime.start_stop_engine import start_engine, stop_engine
from wrappers.action_wrappers import VALID_HEADING_TYPES
import gymnasium as gym
import numpy as np


DEFAULT_OBS_SHAPE = (84, 84)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="RLlib Algorithm checkpoint dir or Policy checkpoint dir",
    )
    parser.add_argument("--policy-id", type=str, default="default_policy", help="policy id inside the Algorithm checkpoint")
    parser.add_argument("--map", type=str, default="_custom_technical_floor", help="evaluation map name under --maps-dir")
    parser.add_argument("--maps-dir", type=str, default="./maps", help="local maps directory")
    parser.add_argument("--episodes", type=int, default=5, help="number of evaluation episodes")
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
        help="optional path for engine stdout/stderr log; default is runs_db21j_multi_engine_rllib/eval_logs/eval_rllib_<map>.log",
    )
    return parser.parse_args()


def resolve_policy_checkpoint(checkpoint_path: Path, policy_id: str) -> Path:
    if (checkpoint_path / "policy_state.pkl").is_file():
        return checkpoint_path

    candidate = checkpoint_path / "policies" / policy_id
    if (candidate / "policy_state.pkl").is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve policy checkpoint from {checkpoint_path}. "
        f"Expected {checkpoint_path / 'policy_state.pkl'} or {candidate / 'policy_state.pkl'}"
    )


def validate_args(args) -> tuple[Path, Path, Path]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
        engine_log_path = (
            Path("./runs_db21j_multi_engine_rllib") / "eval_logs" / f"eval_rllib_{args.map}.log"
        ).resolve()
    engine_log_path.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, maps_root, engine_log_path


def build_env(args) -> gym.Env:
    respawn_kwargs = {
        "lateral_jitter": 0.02,
        "yaw_jitter_deg": 8.0,
        "fallback_bbox": None,
        "avoid_junction": True,
        "max_spawn_angle_deg": 8.0,
    }
    reward_kwargs = {
        "reward_mode": "posangle",
        "include_velocity_reward": True,
        "dist_penalty_alpha": 0.5,
    }
    env = make_single_env(
        entity_name=args.entity_name,
        headless=not args.show_figure,
        max_episode_steps=args.max_episode_steps,
        respawn_mode=args.respawn_mode,
        respawn_backend=args.respawn_backend,
        respawn_kwargs=respawn_kwargs,
        reward_kwargs=reward_kwargs,
        obs_size=DEFAULT_OBS_SHAPE,
        crop_top_ratio=0.33,
        forward_speed=args.forward_speed,
        forward_speed_min=args.forward_speed_min,
        forward_speed_max=args.forward_speed_max,
        max_steer=DEFAULT_MAX_STEER,
        heading_type=args.heading_type,
        frame_repeat_prob=args.frame_repeat_prob,
        motion_blur_kernel_size=args.motion_blur_kernel_size,
        photometric_aug_strength=args.photometric_aug_strength,
        yellow_lane_aug_strength=args.yellow_lane_aug_strength,
        observation_mode=args.observation_mode,
        lane_mask_noise_strength=args.lane_mask_noise_strength,
        engine_host=args.engine_host,
        engine_port=args.engine_port,
    )
    env = ChannelFrameStack(env, n_stack=args.frame_stack)
    return env


def main() -> int:
    args = parse_args()
    apply_dtps_shutdown_patch()
    checkpoint_path, maps_root, engine_log_path = validate_args(args)
    policy_checkpoint = resolve_policy_checkpoint(checkpoint_path, args.policy_id)

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
    engine_respawn_kwargs = {
        "yaw_jitter_deg": 8.0,
        "max_spawn_angle_deg": 8.0,
    }

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
                "DUCKIEMATRIX_RESPAWN_YAW_JITTER_DEG": str(engine_respawn_kwargs["yaw_jitter_deg"]),
                "DUCKIEMATRIX_RESPAWN_MAX_SPAWN_ANGLE_DEG": str(engine_respawn_kwargs["max_spawn_angle_deg"]),
            },
            stdout=engine_log,
            stderr=subprocess.STDOUT,
        )

        env = build_env(args)
        policy = Policy.from_checkpoint(str(policy_checkpoint))
        state = policy.get_initial_state()

        print(f"[INFO] checkpoint={checkpoint_path}")
        print(f"[INFO] policy_checkpoint={policy_checkpoint}")
        print(f"[INFO] map={args.map}")
        print(f"[INFO] engine_log={engine_log_path}")

        episode_rewards = []
        episode_lengths = []
        for episode_idx in range(1, args.episodes + 1):
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

            terminated_reason = "time_limit" if truncated else "terminated"
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
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
