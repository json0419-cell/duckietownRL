#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Main import DEFAULT_MAX_EPISODE_STEPS, DEFAULT_FORWARD_SPEED, discover_maps, make_single_env
from multi_standalone import build_instances, stop_engine_container, terminate_process


DEFAULT_WORLD_PORT = 7501


@dataclass
class InstanceSpec:
    map_name: str
    host: str
    world_port: int
    entity_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one PPO policy against multiple Duckiematrix standalone instances. "
            "By default this script launches the instances automatically."
        )
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=None,
        help="comma-separated map labels; default is all maps under --maps-dir",
    )
    parser.add_argument("--maps-dir", type=str, default="./maps", help="directory containing local maps")
    parser.add_argument(
        "--ports",
        type=str,
        default=None,
        help="comma-separated world/DTPS ports, e.g. 7501,7521,7531; mainly for --attach-only",
    )
    parser.add_argument(
        "--port-offsets",
        type=str,
        default=None,
        help="comma-separated port offsets; used for auto-launch or if --ports is omitted",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="engine host for all instances")
    parser.add_argument("--entity-name", type=str, default="map_0/vehicle_0", help="target vehicle entity name")
    parser.add_argument(
        "--attach-only",
        action="store_true",
        help="do not launch standalone processes; attach to already-running instances instead",
    )
    parser.add_argument(
        "--engine-name-prefix",
        type=str,
        default="dts-matrix-engine",
        help="engine container name prefix for auto-launched instances",
    )
    parser.add_argument("--sandbox", action="store_true", help="pass --sandbox when auto-launching standalone instances")
    parser.add_argument(
        "--graphics-api",
        type=str,
        default="opengl",
        choices=("opengl", "vulkan", "default"),
        help="renderer graphics API for auto-launched standalone instances",
    )
    parser.add_argument("--pull", action="store_true", help="allow dts to pull instead of using --no-pull")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="total PPO timesteps")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="TimeLimit wrapper; use 0 to disable",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="PPO learning rate")
    parser.add_argument("--device", type=str, default="auto", help="torch device")
    parser.add_argument("--logdir", type=str, default="./runs_db21j_multi_engine", help="output directory")
    parser.add_argument(
        "--standalone-logdir",
        type=str,
        default=None,
        help="log directory for auto-launched standalone processes; default is <logdir>/standalone_logs",
    )
    parser.add_argument("--save-name", type=str, default="ppo_db21j_multi_engine", help="model basename")
    parser.add_argument("--load-model", type=str, default=None, help="resume from a PPO .zip")
    parser.add_argument("--respawn-mode", type=str, default="random", choices=("random", "fixed"))
    parser.add_argument(
        "--respawn-backend",
        type=str,
        default="engine",
        choices=("engine", "wrapper", "hybrid"),
        help="which side owns respawn selection/validation",
    )
    parser.add_argument("--headless", action="store_true", help="leave DB21J matplotlib windows disabled")
    parser.add_argument(
        "--start-method",
        type=str,
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
        help="SubprocVecEnv start method",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="save checkpoint every N timesteps; 0 disables periodic checkpoints",
    )
    return parser.parse_args()


def parse_maps(raw: str) -> list[str]:
    maps = [m.strip() for m in raw.split(",") if m.strip()]
    if not maps:
        raise ValueError("No maps specified")
    return maps


def resolve_maps(args: argparse.Namespace) -> list[str]:
    if args.maps:
        return parse_maps(args.maps)
    maps_root = Path(args.maps_dir).expanduser().resolve()
    return discover_maps(maps_root)


def default_offsets(count: int) -> list[int]:
    offsets = [0]
    for idx in range(1, count):
        offsets.append((idx + 1) * 10)
    return offsets


def parse_ports_or_offsets(args: argparse.Namespace, count: int) -> list[int]:
    if args.ports:
        ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
        if len(ports) != count:
            raise ValueError(f"--ports expects {count} values, got {len(ports)}")
        return ports

    if args.port_offsets:
        offsets = [int(v.strip()) for v in args.port_offsets.split(",") if v.strip()]
        if len(offsets) != count:
            raise ValueError(f"--port-offsets expects {count} values, got {len(offsets)}")
    else:
        offsets = default_offsets(count)

    return [DEFAULT_WORLD_PORT + offset for offset in offsets]


def check_url_nonempty(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            raw = resp.read()
        return bool(raw)
    except Exception:
        return False


def wait_instance_ready(spec: InstanceSpec, timeout_s: float = 40.0) -> None:
    pose_url = f"http://{spec.host}:{spec.world_port}/robot/{spec.entity_name}/state/pose/"
    cam_url = f"http://{spec.host}:{spec.world_port}/robot/{spec.entity_name}/sensor/camera/front_center/jpeg/"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if check_url_nonempty(pose_url) and check_url_nonempty(cam_url):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Instance not ready on port {spec.world_port}: {pose_url}")


def make_env_factory(
    spec: InstanceSpec,
    *,
    max_episode_steps: int,
    respawn_mode: str,
    respawn_backend: str,
    reward_kwargs: dict,
    respawn_kwargs: dict,
):
    def _factory():
        return make_single_env(
            entity_name=spec.entity_name,
            headless=True,
            max_episode_steps=max_episode_steps,
            respawn_mode=respawn_mode,
            respawn_backend=respawn_backend,
            respawn_kwargs=respawn_kwargs,
            reward_kwargs=reward_kwargs,
            obs_size=(80, 160),
            crop_top_ratio=0.33,
            forward_speed=DEFAULT_FORWARD_SPEED,
            max_steer=1.0,
            engine_host=spec.host,
            engine_port=spec.world_port,
        )

    return _factory


def build_multi_vec_env(
    specs: list[InstanceSpec],
    *,
    max_episode_steps: int,
    respawn_mode: str,
    respawn_backend: str,
    reward_kwargs: dict,
    respawn_kwargs: dict,
    start_method: str,
):
    factories = [
        make_env_factory(
            spec,
            max_episode_steps=max_episode_steps,
            respawn_mode=respawn_mode,
            respawn_backend=respawn_backend,
            reward_kwargs=reward_kwargs,
            respawn_kwargs=respawn_kwargs,
        )
        for spec in specs
    ]
    if len(factories) == 1:
        venv = DummyVecEnv(factories)
    else:
        venv = SubprocVecEnv(factories, start_method=start_method)
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=3)
    return venv


def auto_launch_instances(
    args: argparse.Namespace,
    maps: list[str],
    logdir: Path,
    *,
    env_overrides: dict[str, str] | None = None,
):
    standalone_logdir = (
        Path(args.standalone_logdir).expanduser().resolve()
        if args.standalone_logdir
        else (logdir / "standalone_logs").resolve()
    )
    launcher_args = argparse.Namespace(
        maps=",".join(maps),
        maps_dir=args.maps_dir,
        logdir=str(standalone_logdir),
        engine_name_prefix=args.engine_name_prefix,
        port_offsets=args.port_offsets,
        sandbox=args.sandbox,
        graphics_api=args.graphics_api,
        pull=args.pull,
        dry_run=False,
    )
    instances = build_instances(launcher_args)
    launched: list[tuple[object, subprocess.Popen, object]] = []
    for inst in instances:
        stop_engine_container(inst.engine_name)
        log_file = open(inst.log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.Popen(
            inst.cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        launched.append((inst, proc, log_file))
        print(
            f"[STARTED] map={inst.map_label} engine={inst.engine_name} "
            f"world_port={inst.world_port} log={inst.log_path}"
        )
        time.sleep(2.0)
        if proc.poll() is not None:
            raise RuntimeError(
                f"Standalone for map={inst.map_label} exited early with code {proc.returncode}. "
                f"See {inst.log_path}"
            )
    return instances, launched


def main() -> int:
    args = parse_args()
    logdir = Path(args.logdir).expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    maps = resolve_maps(args)
    launched_instances: list[tuple[object, subprocess.Popen, object]] = []
    respawn_kwargs = {
        "lateral_jitter": 0.02,
        "yaw_jitter_deg": 0.0,
        "fallback_bbox": None,
        "avoid_junction": True,
        "max_spawn_angle_deg": 4.0,
    }
    engine_respawn_mode = args.respawn_mode if args.respawn_backend != "wrapper" else "fixed"
    engine_env_overrides = {
        "DUCKIEMATRIX_RESPAWN_MODE": engine_respawn_mode,
        "DUCKIEMATRIX_RESPAWN_MAX_SPAWN_ANGLE_DEG": str(respawn_kwargs["max_spawn_angle_deg"]),
    }

    if args.attach_only:
        world_ports = parse_ports_or_offsets(args, len(maps))
        specs = [
            InstanceSpec(
                map_name=map_name,
                host=args.host,
                world_port=world_port,
                entity_name=args.entity_name,
            )
            for map_name, world_port in zip(maps, world_ports)
        ]
    else:
        instances, launched_instances = auto_launch_instances(
            args,
            maps,
            logdir,
            env_overrides=engine_env_overrides,
        )
        specs = [
            InstanceSpec(
                map_name=inst.map_label,
                host=args.host,
                world_port=inst.world_port,
                entity_name=args.entity_name,
            )
            for inst in instances
        ]

    for spec in specs:
        print(
            f"[PLAN] map={spec.map_name} host={spec.host} world_port={spec.world_port} "
            f"entity={spec.entity_name}"
        )
        wait_instance_ready(spec)

    reward_kwargs = {
        "reward_mode": "posangle",
        "include_velocity_reward": True,
    }

    venv = build_multi_vec_env(
        specs,
        max_episode_steps=args.max_episode_steps,
        respawn_mode=args.respawn_mode,
        respawn_backend=args.respawn_backend,
        reward_kwargs=reward_kwargs,
        respawn_kwargs=respawn_kwargs,
        start_method=args.start_method,
    )

    logger = configure(str(logdir), ["stdout", "csv", "tensorboard"])

    model_path = Path(args.load_model).expanduser().resolve() if args.load_model else None
    try:
        if model_path is not None:
            if not model_path.is_file():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"[INFO] resuming from: {model_path}")
            model = PPO.load(str(model_path), env=venv, device=args.device)
            model.learning_rate = args.learning_rate
            model.lr_schedule = get_schedule_fn(args.learning_rate)
            for group in model.policy.optimizer.param_groups:
                group["lr"] = float(args.learning_rate)
        else:
            model = PPO(
                policy="CnnPolicy",
                env=venv,
                verbose=1,
                tensorboard_log=str(logdir),
                n_steps=1024,
                batch_size=64,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=args.learning_rate,
                clip_range=0.2,
                target_kl=0.05,
                device=args.device,
            )

        model.set_logger(logger)

        callbacks = []
        if args.checkpoint_freq > 0:
            save_freq = max(args.checkpoint_freq // max(len(specs), 1), 1)
            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(logdir),
                    name_prefix=args.save_name,
                )
            )

        callback = callbacks[0] if len(callbacks) == 1 else callbacks or None
        print(f"[INFO] training with {len(specs)} engines for {args.timesteps} timesteps")
        model.learn(total_timesteps=args.timesteps, callback=callback)
        final_path = logdir / f"{args.save_name}.zip"
        model.save(str(final_path))
        print(f"[DONE] saved model: {final_path}")
        return 0
    finally:
        venv.close()
        for inst, proc, log_file in reversed(launched_instances):
            terminate_process(proc)
            stop_engine_container(inst.engine_name)
            try:
                log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
