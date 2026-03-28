import argparse
import os
import random
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import cbor2
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from action_wrappers import HeadingToWheelsWrapper
from duckiematrix_env import DuckiematrixDB21JEnv
from dtps_shutdown_patch import apply_dtps_shutdown_patch
from map_interpreter_patch import use_patched_map_interpreter
from observation_wrappers import ResizeCropWrapper
from respawn_wrapper import VALID_RESPAWN_MODES, maybe_wrap_respawn
from reward_wrappers import LaneFollowingRewardWrapper
from start_stop_engine import start_engine, stop_engine


DEFAULT_TOTAL_TIMESTEPS = 1_500_000
DEFAULT_SEGMENT_TIMESTEPS = 16_384
DEFAULT_MAX_EPISODE_STEPS = 300
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_FORWARD_SPEED = 1.0


def close_vec_env_gracefully(venv, drain_s: float = 0.2) -> None:
    try:
        venv.close()
    finally:
        time.sleep(max(0.0, float(drain_s)))


def _read_topic_timestamp(url: str, timeout_s: float = 2.0) -> float | None:
    try:
        raw = urllib.request.urlopen(url, timeout=timeout_s).read()
        msg = cbor2.loads(raw)
        return float(msg["header"]["timestamp"])
    except Exception:
        return None


class CameraFreezeCallback(BaseCallback):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        entity_name: str,
        lag_threshold_s: float,
        poll_every_steps: int,
        start_after_steps: int = 256,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.pose_url = f"http://{host}:{port}/robot/{entity_name}/state/pose/"
        self.cam_url = f"http://{host}:{port}/robot/{entity_name}/sensor/camera/front_center/jpeg/"
        self.lag_threshold_s = float(lag_threshold_s)
        self.poll_every_steps = max(1, int(poll_every_steps))
        self.start_after_steps = max(0, int(start_after_steps))
        self.freeze_detected = False
        self.freeze_reason = ""
        self.last_pose_ts = None
        self.last_cam_ts = None

    def _on_step(self) -> bool:
        if self.n_calls < self.start_after_steps:
            return True
        if self.n_calls % self.poll_every_steps != 0:
            return True

        pose_ts = _read_topic_timestamp(self.pose_url)
        cam_ts = _read_topic_timestamp(self.cam_url)
        self.last_pose_ts = pose_ts
        self.last_cam_ts = cam_ts

        if pose_ts is None or cam_ts is None:
            self.freeze_detected = True
            self.freeze_reason = f"camera watchdog read failed: pose_ts={pose_ts} cam_ts={cam_ts}"
            return False

        lag = float(pose_ts) - float(cam_ts)
        if lag > self.lag_threshold_s:
            self.freeze_detected = True
            self.freeze_reason = (
                f"camera stream frozen: pose_ts={pose_ts:.3f} cam_ts={cam_ts:.3f} lag={lag:.3f}s "
                f"threshold={self.lag_threshold_s:.3f}s"
            )
            return False
        return True


def make_single_env(
    entity_name: str = "map_0/vehicle_0",
    headless: bool = True,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    respawn_mode: str = "random",
    respawn_kwargs: dict | None = None,
    reward_kwargs: dict | None = None,
    obs_size: tuple[int, int] = (80, 160),
    crop_top_ratio: float = 0.33,
    forward_speed: float = DEFAULT_FORWARD_SPEED,
    max_steer: float = 1.0,
    engine_host: str | None = None,
    engine_port: int | None = None,
):
    env = DuckiematrixDB21JEnv(
        entity_name=entity_name,
        out_of_road_penalty=-10.0,
        headless=headless,
        camera_height=480,
        camera_width=640,
        host=engine_host,
        port=engine_port,
    )
    use_patched_map_interpreter(env)

    env = maybe_wrap_respawn(
        env,
        respawn_mode=respawn_mode,
        respawn_kwargs=respawn_kwargs,
    )

    reward_kwargs = reward_kwargs or {}
    env = LaneFollowingRewardWrapper(env, **reward_kwargs)

    out_h, out_w = obs_size
    env = ResizeCropWrapper(env, out_h=out_h, out_w=out_w, crop_top_ratio=crop_top_ratio)
    env = HeadingToWheelsWrapper(
        env,
        forward_speed=forward_speed,
        max_steer=max_steer,
    )
    if max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def build_vec_env(
    *,
    entity_name: str,
    headless: bool,
    max_episode_steps: int,
    respawn_mode: str,
    respawn_kwargs: dict,
    reward_kwargs: dict,
    engine_host: str | None = None,
    engine_port: int | None = None,
):
    def _factory():
        return make_single_env(
            entity_name=entity_name,
            headless=headless,
            max_episode_steps=max_episode_steps,
            respawn_mode=respawn_mode,
            respawn_kwargs=respawn_kwargs,
            reward_kwargs=reward_kwargs,
            obs_size=(80, 160),
            crop_top_ratio=0.33,
            forward_speed=DEFAULT_FORWARD_SPEED,
            max_steer=1.0,
            engine_host=engine_host,
            engine_port=engine_port,
        )

    venv = DummyVecEnv([_factory])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=3)
    return venv


def discover_maps(maps_root: Path) -> list[str]:
    maps = []
    for child in sorted(maps_root.iterdir()):
        if child.is_dir() and (child / "main.yaml").exists():
            maps.append(child.name)
    if not maps:
        raise RuntimeError(f"No maps found under {maps_root}")
    return maps


def parse_map_subset_arg(map_subset: str | None) -> list[str] | None:
    if map_subset is None:
        return None
    names = [part.strip() for part in str(map_subset).split(",")]
    names = [name for name in names if name]
    if not names:
        raise RuntimeError("map-subset was provided but no valid map names were found.")
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def map_engine_arg(maps_dir_arg: str, map_name: str) -> str:
    return str(Path(maps_dir_arg) / map_name)


def build_schedule(
    available_maps: list[str],
    total_timesteps: int,
    segment_timesteps: int,
    *,
    first_map: str,
    rng: random.Random,
    map_order: str,
) -> list[tuple[str, int]]:
    if first_map not in available_maps:
        raise RuntimeError(f"Required first map '{first_map}' not found in maps directory.")
    if total_timesteps <= 0 or segment_timesteps <= 0:
        raise ValueError("total_timesteps and segment_timesteps must be positive.")

    schedule: list[tuple[str, int]] = []
    remaining = int(total_timesteps)
    segment_idx = 0
    ordered_maps = [first_map] + [m for m in available_maps if m != first_map]
    while remaining > 0:
        steps = min(int(segment_timesteps), remaining)
        if segment_idx == 0:
            map_name = first_map
        elif map_order == "round_robin":
            map_name = ordered_maps[segment_idx % len(ordered_maps)]
        else:
            map_name = rng.choice(available_maps)
        schedule.append((map_name, steps))
        remaining -= steps
        segment_idx += 1
    return schedule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, help="total timesteps to train")
    p.add_argument(
        "--segment-timesteps",
        type=int,
        default=DEFAULT_SEGMENT_TIMESTEPS,
        help="timesteps to train on one map before restarting the engine",
    )
    p.add_argument("--logdir", type=str, default="./runs_db21j_ppo", help="logging/checkpoint dir")
    p.add_argument("--entity-name", type=str, default="map_0/vehicle_0", help="target vehicle entity name")
    p.add_argument(
        "--show-figure",
        action="store_true",
        help="show the local matplotlib figure window from DB21JEnv (disabled by default)",
    )
    p.add_argument("--save-name", type=str, default="ppo_db21j_final", help="final model filename")
    p.add_argument("--load-model", type=str, default=None, help="path to existing PPO model .zip to resume training")
    p.add_argument("--device", type=str, default="auto", help="torch device: auto/cpu/cuda")
    p.add_argument(
        "--respawn-mode",
        type=str,
        default="random",
        choices=VALID_RESPAWN_MODES,
        help="respawn mode shared by the training wrapper and the engine patch",
    )
    p.add_argument("--maps-dir", type=str, default="./maps", help="local maps directory")
    p.add_argument(
        "--only-map",
        type=str,
        default=None,
        help="restrict training to a single map inside --maps-dir; disables map rotation",
    )
    p.add_argument(
        "--map-subset",
        type=str,
        default=None,
        help="comma-separated map subset inside --maps-dir; preserves the given order",
    )
    p.add_argument("--first-map", type=str, default="loop", help="map used for the first training segment")
    p.add_argument("--seed", type=int, default=1234, help="random seed for map selection")
    p.add_argument(
        "--map-order",
        type=str,
        default="round_robin",
        choices=("round_robin", "random"),
        help="map schedule for single-engine training after the first segment",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="PPO learning rate; defaults to Duckietown-RL PPO lr",
    )
    p.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS, help="TimeLimit wrapper")
    p.add_argument("--engine-host", type=str, default="127.0.0.1", help="engine host")
    p.add_argument("--engine-port", type=int, default=7501, help="engine DTPS port")
    p.add_argument("--engine-ready-timeout", type=float, default=40.0, help="wait engine readiness timeout")
    p.add_argument("--container-name", type=str, default="dts-matrix-engine", help="engine docker container name")
    p.add_argument(
        "--graphics-api",
        type=str,
        default="opengl",
        choices=("opengl", "vulkan", "default"),
        help="renderer graphics API for standalone mode",
    )
    p.add_argument(
        "--renderer-process-name",
        type=str,
        default="duckiematrix.x86_64",
        help="process name pattern used to stop the local renderer",
    )
    p.add_argument(
        "--camera-freeze-threshold-s",
        type=float,
        default=3.0,
        help="abort training if pose_ts - camera_ts exceeds this threshold",
    )
    p.add_argument(
        "--camera-watchdog-steps",
        type=int,
        default=128,
        help="poll camera/pose timestamps every N environment steps",
    )
    p.add_argument("--pull", action="store_true", help="allow dts to pull instead of using --no-pull")
    return p.parse_args()


def main():
    args = parse_args()
    apply_dtps_shutdown_patch()
    logdir = Path(args.logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    engine_log_dir = logdir / "engine_logs"
    engine_log_dir.mkdir(parents=True, exist_ok=True)

    maps_root = Path(args.maps_dir).resolve()
    available_maps = discover_maps(maps_root)
    if args.only_map is not None:
        if args.only_map not in available_maps:
            raise RuntimeError(f"Requested only-map '{args.only_map}' not found under {maps_root}")
        available_maps = [args.only_map]
        args.first_map = args.only_map
    elif args.map_subset is not None:
        subset = parse_map_subset_arg(args.map_subset)
        missing = [name for name in subset if name not in available_maps]
        if missing:
            raise RuntimeError(f"Requested map-subset contains unknown maps under {maps_root}: {missing}")
        available_maps = subset
        if args.first_map not in available_maps:
            args.first_map = available_maps[0]
    rng = random.Random(args.seed)
    schedule = build_schedule(
        available_maps,
        args.timesteps,
        args.segment_timesteps,
        first_map=args.first_map,
        rng=rng,
        map_order=args.map_order,
    )

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
        "include_collision_avoidance": True,
    }

    print(f"[INFO] discovered maps: {available_maps}")
    print(
        f"[INFO] training schedule segments={len(schedule)} first={schedule[0][0]} "
        f"segment_steps={args.segment_timesteps} map_order={args.map_order}"
    )

    model = None
    logger = configure(str(logdir), ["stdout", "csv", "tensorboard"])
    current_proc = None
    current_venv = None
    current_engine_log = None

    def _cleanup(*_):
        nonlocal current_proc, current_venv, current_engine_log
        if current_venv is not None:
            try:
                close_vec_env_gracefully(current_venv)
            except Exception:
                pass
            current_venv = None
        if current_proc is not None:
            stop_engine(
                current_proc,
                args.container_name,
                stop_renderer=True,
                renderer_process_name=args.renderer_process_name,
            )
            current_proc = None
        if current_engine_log is not None:
            try:
                current_engine_log.close()
            except Exception:
                pass
            current_engine_log = None

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        for segment_idx, (map_name, segment_steps) in enumerate(schedule, start=1):
            map_arg = map_engine_arg(args.maps_dir, map_name)
            remaining_steps = int(segment_steps)
            segment_attempt = 0

            while remaining_steps > 0:
                segment_attempt += 1
                print(
                    f"\n[SEGMENT {segment_idx:02d}/{len(schedule)}] "
                    f"attempt={segment_attempt} map={map_name} engine_map={map_arg} remaining_timesteps={remaining_steps}"
                )

                engine_log_name = f"segment_{segment_idx:02d}_{map_name}_attempt_{segment_attempt:02d}.log"
                engine_log_path = engine_log_dir / engine_log_name
                current_engine_log = open(engine_log_path, "w", encoding="utf-8")

                current_proc, _, _ = start_engine(
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
                    stdout=current_engine_log,
                    stderr=subprocess.STDOUT,
                )

                current_venv = build_vec_env(
                    entity_name=args.entity_name,
                    headless=not args.show_figure,
                    max_episode_steps=args.max_episode_steps,
                    respawn_mode=args.respawn_mode,
                    respawn_kwargs=respawn_kwargs,
                    reward_kwargs=reward_kwargs,
                    engine_host=args.engine_host,
                    engine_port=args.engine_port,
                )

                if model is None:
                    if args.load_model:
                        print(f"[INFO] resuming training from: {args.load_model}")
                        model = PPO.load(args.load_model, env=current_venv, device=args.device)
                        model.learning_rate = args.learning_rate
                        model.lr_schedule = get_schedule_fn(args.learning_rate)
                        for group in model.policy.optimizer.param_groups:
                            group["lr"] = float(args.learning_rate)
                    else:
                        model = PPO(
                            policy="CnnPolicy",
                            env=current_venv,
                            verbose=1,
                            tensorboard_log=str(logdir),
                            n_steps=2048,
                            batch_size=64,
                            n_epochs=5,
                            gamma=0.99,
                            gae_lambda=0.95,
                            learning_rate=args.learning_rate,
                            clip_range=0.2,
                            target_kl=0.03,
                            device=args.device,
                        )
                    model.set_logger(logger)
                else:
                    model.set_env(current_venv, force_reset=True)

                camera_watchdog = CameraFreezeCallback(
                    host=args.engine_host,
                    port=args.engine_port,
                    entity_name=args.entity_name,
                    lag_threshold_s=args.camera_freeze_threshold_s,
                    poll_every_steps=args.camera_watchdog_steps,
                    start_after_steps=max(256, args.camera_watchdog_steps),
                )
                before_timesteps = int(model.num_timesteps)
                model.learn(
                    total_timesteps=remaining_steps,
                    reset_num_timesteps=(segment_idx == 1 and segment_attempt == 1 and args.load_model is None),
                    callback=camera_watchdog,
                )
                progressed = int(model.num_timesteps) - before_timesteps
                remaining_steps = max(0, remaining_steps - max(progressed, 0))

                close_vec_env_gracefully(current_venv)
                current_venv = None
                stop_engine(
                    current_proc,
                    args.container_name,
                    stop_renderer=True,
                    renderer_process_name=args.renderer_process_name,
                )
                current_proc = None
                current_engine_log.close()
                current_engine_log = None

                if camera_watchdog.freeze_detected:
                    raise RuntimeError(
                        "Camera freeze detected; aborting training: "
                        f"{camera_watchdog.freeze_reason}; progressed={progressed}; remaining={remaining_steps}"
                    )

                break

            if remaining_steps > 0:
                raise RuntimeError(
                    f"Segment {segment_idx} on map '{map_name}' ended early with remaining_steps={remaining_steps}."
                )

            segment_save_name = logdir / f"ppo_db21j_seg{segment_idx:02d}_{map_name}"
            model.save(str(segment_save_name))
            print(f"[INFO] saved segment model: {segment_save_name}.zip")

        final_path = logdir / args.save_name
        model.save(str(final_path))
        print(f"\n[DONE] final model saved to: {final_path}.zip")
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
