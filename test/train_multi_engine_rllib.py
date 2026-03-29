#!/usr/bin/env python3
import argparse
import pickle
import subprocess
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Main import DEFAULT_FORWARD_SPEED, DEFAULT_MAX_EPISODE_STEPS, discover_maps, make_single_env
from multi_standalone import build_instances, stop_engine_container, terminate_process


DEFAULT_WORLD_PORT = 7501
DEFAULT_OBS_SHAPE = (84, 84)
DEFAULT_MODEL_CONFIG = {
    # RLlib has no built-in CNN template for 84x84x9 (3 stacked RGB frames).
    # The old VisionNet uses SAME padding for all but the last Conv2D, so this
    # stack reduces 84x84 -> 21x21 -> 11x11 -> 1x1.
    "conv_filters": [
        [16, [8, 8], 4],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ],
    "conv_activation": "relu",
    "fcnet_hiddens": [256],
    "fcnet_activation": "relu",
}


@dataclass
class InstanceSpec:
    map_name: str
    host: str
    world_port: int
    entity_name: str


class ChannelFrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, n_stack: int = 3):
        super().__init__(env)
        self.n_stack = max(1, int(n_stack))
        self.frames = deque(maxlen=self.n_stack)

        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("ChannelFrameStack expects a Box observation space")
        if len(obs_space.shape) != 3:
            raise ValueError(f"ChannelFrameStack expects HWC obs, got shape={obs_space.shape}")

        h, w, c = obs_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(h, w, c * self.n_stack),
            dtype=np.float32,
        )


    @staticmethod
    def _prepare_frame(obs) -> np.ndarray:
        frame = np.asarray(obs, dtype=np.float32)
        if frame.max() > 1.0:
            frame = frame / 255.0
        return frame

    def _stacked_obs(self):
        return np.concatenate(list(self.frames), axis=2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._prepare_frame(obs)
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(obs.copy())
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._prepare_frame(obs).copy())
        return self._stacked_obs(), reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one RLlib PPO policy against multiple Duckiematrix standalone instances. "
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
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="total RLlib timesteps")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="TimeLimit wrapper; use 0 to disable",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="PPO learning rate")
    parser.add_argument("--logdir", type=str, default="./runs_db21j_multi_engine_rllib", help="output directory")
    parser.add_argument(
        "--standalone-logdir",
        type=str,
        default=None,
        help="log directory for auto-launched standalone processes; default is <logdir>/standalone_logs",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="save checkpoint every N timesteps; 0 disables periodic checkpoints",
    )
    parser.add_argument("--save-name", type=str, default="rllib_db21j_multi_engine", help="checkpoint basename")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="resume from an RLlib checkpoint directory")
    parser.add_argument("--respawn-mode", type=str, default="random", choices=("random", "fixed"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "number of remote rollout workers; 0 means one worker per requested map. "
            "If larger than the map count, maps are repeated modulo-style."
        ),
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=str,
        default="auto",
        help="per-worker fragment length; use 'auto' to let RLlib choose a compatible value",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4096,
        help="RLlib train batch size, matching Duckietown-RL default",
    )
    parser.add_argument(
        "--sgd-minibatch-size",
        type=int,
        default=128,
        help="RLlib SGD minibatch size, matching Duckietown-RL default",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=3,
        help="number of RGB frames stacked along channels before feeding RLlib",
    )
    parser.add_argument(
        "--sample-timeout-s",
        type=float,
        default=300.0,
        help="time to wait for remote env runners to return samples before warning",
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


def expand_maps_for_workers(maps: list[str], num_workers: int) -> list[str]:
    if not maps:
        raise ValueError("No maps available to expand")
    if num_workers <= 0 or num_workers == len(maps):
        return list(maps)
    return [maps[idx % len(maps)] for idx in range(num_workers)]


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


def auto_launch_instances(args: argparse.Namespace, maps: list[str], logdir: Path):
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
        proc = subprocess.Popen(
            inst.cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
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


def make_rllib_env(env_config):
    worker_index = int(getattr(env_config, "worker_index", env_config.get("worker_index", 0)))
    specs = env_config["specs"]
    if env_config.get("has_local_env_runner", False):
        spec_idx = worker_index % len(specs)
    else:
        spec_idx = (max(worker_index, 1) - 1) % len(specs)
    spec = specs[spec_idx]

    env = make_single_env(
        entity_name=spec["entity_name"],
        headless=True,
        max_episode_steps=int(env_config["max_episode_steps"]),
        respawn_mode=str(env_config["respawn_mode"]),
        respawn_kwargs=dict(env_config["respawn_kwargs"]),
        reward_kwargs=dict(env_config["reward_kwargs"]),
        obs_size=DEFAULT_OBS_SHAPE,
        crop_top_ratio=0.33,
        forward_speed=DEFAULT_FORWARD_SPEED,
        max_steer=1.0,
        engine_host=spec["host"],
        engine_port=int(spec["world_port"]),
    )
    env = ChannelFrameStack(env, n_stack=int(env_config["frame_stack"]))
    return env


def parse_rollout_fragment_length(raw: str) -> int | str:
    text = str(raw).strip().lower()
    if text == "auto":
        return "auto"
    value = int(text)
    if value <= 0:
        raise ValueError("--rollout-fragment-length must be positive or 'auto'")
    return value


def _result_timesteps(result: dict) -> int:
    for key in (
        "timesteps_total",
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "agent_timesteps_total",
    ):
        value = result.get(key)
        if value is not None:
            return int(value)

    env_runners = result.get("env_runners", {})
    for key in ("num_env_steps_sampled_lifetime", "num_env_steps_sampled"):
        value = env_runners.get(key)
        if value is not None:
            return int(value)
    raise KeyError("Could not find cumulative timesteps in RLlib result dict")


def _result_reward_mean(result: dict) -> float | None:
    value = result.get("episode_reward_mean")
    if value is not None:
        return float(value)
    env_runners = result.get("env_runners", {})
    for key in ("episode_return_mean", "episode_reward_mean"):
        value = env_runners.get(key)
        if value is not None:
            return float(value)
    return None


def _save_checkpoint(algo, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        saved = algo.save(checkpoint_dir=str(save_dir))
    except TypeError:
        saved = algo.save(str(save_dir))

    if hasattr(saved, "checkpoint") and hasattr(saved.checkpoint, "path"):
        return Path(saved.checkpoint.path).resolve()
    return Path(str(saved)).resolve()


def _checkpoint_resume_timesteps(checkpoint_dir: Path) -> int:
    state_file = checkpoint_dir / "algorithm_state.pkl"
    if not state_file.exists():
        return 0

    with state_file.open("rb") as f:
        state = pickle.load(f)

    if not isinstance(state, dict):
        return 0

    counters = state.get("counters", {})
    for key in (
        "num_env_steps_sampled",
        "num_env_steps_sampled_lifetime",
        "num_agent_steps_sampled",
        "num_agent_steps_sampled_lifetime",
    ):
        value = counters.get(key)
        if value is not None:
            return int(value)
    return 0


def main() -> int:
    args = parse_args()

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except Exception as e:
        raise RuntimeError(
            "RLlib is not installed in the current environment. "
            "Install a Ray build that includes RLlib before running this script."
        ) from e

    logdir = Path(args.logdir).expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    maps = resolve_maps(args)
    worker_maps = expand_maps_for_workers(maps, args.num_workers)
    launched_instances: list[tuple[object, subprocess.Popen, object]] = []

    if args.attach_only:
        world_ports = parse_ports_or_offsets(args, len(worker_maps))
        specs = [
            InstanceSpec(
                map_name=map_name,
                host=args.host,
                world_port=world_port,
                entity_name=args.entity_name,
            )
            for map_name, world_port in zip(worker_maps, world_ports)
        ]
    else:
        instances, launched_instances = auto_launch_instances(args, worker_maps, logdir)
        specs = [
            InstanceSpec(
                map_name=inst.map_label,
                host=args.host,
                world_port=inst.world_port,
                entity_name=args.entity_name,
            )
            for inst in instances
        ]

    try:
        for spec in specs:
            print(
                f"[PLAN] map={spec.map_name} host={spec.host} world_port={spec.world_port} "
                f"entity={spec.entity_name}"
            )
            wait_instance_ready(spec)

        total_workers = len(specs)
        rollout_fragment_length = parse_rollout_fragment_length(args.rollout_fragment_length)

        reward_kwargs = {
            "reward_mode": "posangle",
            "include_velocity_reward": True,
        }
        respawn_kwargs = {
            "lateral_jitter": 0.02,
            "yaw_jitter_deg": 0.0,
            "fallback_bbox": None,
            "avoid_junction": True,
            "max_spawn_angle_deg": 4.0,
        }
        env_config = {
            "specs": [
                {
                    "map_name": spec.map_name,
                    "host": spec.host,
                    "world_port": spec.world_port,
                    "entity_name": spec.entity_name,
                }
                for spec in specs
            ],
            "max_episode_steps": args.max_episode_steps,
            "respawn_mode": args.respawn_mode,
            "respawn_kwargs": respawn_kwargs,
            "reward_kwargs": reward_kwargs,
            "frame_stack": args.frame_stack,
            "has_local_env_runner": False,
        }

        prototype_env = make_rllib_env(
            {
                **env_config,
                "worker_index": 0,
                "has_local_env_runner": True,
            }
        )
        observation_space = prototype_env.observation_space
        action_space = prototype_env.action_space
        prototype_env.close()

        env_name = "duckiematrix_multi_engine_rllib_env"
        register_env(env_name, make_rllib_env)

        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env=env_name,
                env_config=env_config,
                observation_space=observation_space,
                action_space=action_space,
                disable_env_checking=True,
            )
            .framework("torch")
            .env_runners(
                num_env_runners=total_workers,
                num_envs_per_env_runner=1,
                create_local_env_runner=False,
                rollout_fragment_length=rollout_fragment_length,
                batch_mode="truncate_episodes",
                sample_timeout_s=args.sample_timeout_s,
            )
            .resources(num_gpus=1)
            .training(
                train_batch_size=args.train_batch_size,
                lr=args.learning_rate,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                minibatch_size=args.sgd_minibatch_size,
                vf_loss_coeff=0.5,
                entropy_coeff=0.0,
                grad_clip=0.5,
                model=DEFAULT_MODEL_CONFIG,
            )
            .debugging(seed=args.seed, log_level="INFO")
        )

        algo = config.build_algo()
        start_timesteps = 0
        if args.load_checkpoint:
            ckpt = Path(args.load_checkpoint).expanduser().resolve()
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            print(f"[INFO] restoring checkpoint: {ckpt}")
            start_timesteps = _checkpoint_resume_timesteps(ckpt)
            algo.restore(str(ckpt))

        checkpoints_dir = logdir / "checkpoints"
        next_checkpoint_t = start_timesteps + max(args.checkpoint_freq, 0)
        best_reward = None
        best_path = None
        target_timesteps = start_timesteps + args.timesteps

        print(
            f"[INFO] RLlib training with {total_workers} remote rollout workers "
            f"(local=0 remote={total_workers}) across {len(specs)} engines "
            f"for {args.timesteps} additional timesteps "
            f"(resume={start_timesteps}, target={target_timesteps})"
        )

        total_timesteps = start_timesteps
        while total_timesteps < target_timesteps:
            result = algo.train()
            total_timesteps = _result_timesteps(result)
            reward_mean = _result_reward_mean(result)
            print(
                f"[TRAIN] iter={result.get('training_iteration')} "
                f"timesteps={total_timesteps} "
                f"reward_mean={reward_mean}"
            )

            if reward_mean is not None and (best_reward is None or reward_mean > best_reward):
                best_reward = reward_mean
                best_dir = checkpoints_dir / f"{args.save_name}_best"
                best_path = _save_checkpoint(algo, best_dir)
                print(f"[BEST] reward_mean={best_reward:.4f} checkpoint={best_path}")

            if args.checkpoint_freq > 0 and total_timesteps >= next_checkpoint_t:
                ckpt_dir = checkpoints_dir / f"{args.save_name}_{total_timesteps}"
                ckpt_path = _save_checkpoint(algo, ckpt_dir)
                print(f"[CHECKPOINT] {ckpt_path}")
                next_checkpoint_t += args.checkpoint_freq

        final_dir = checkpoints_dir / f"{args.save_name}_final"
        final_path = _save_checkpoint(algo, final_dir)
        print(f"[DONE] saved final checkpoint: {final_path}")
        if best_path is not None:
            print(f"[DONE] best checkpoint: {best_path}")
        return 0
    finally:
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass

        for inst, proc, log_file in reversed(launched_instances):
            terminate_process(proc)
            stop_engine_container(inst.engine_name)
            try:
                log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
