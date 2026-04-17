from gymnasium.wrappers import TimeLimit

from wrappers.action_wrappers import HeadingToWheelsWrapper
from wrappers.observation_wrappers import (
    BinaryLaneMaskWrapper,
    DEFAULT_LANE_MASK_NOISE_STRENGTH,
    LaneMaskNoiseWrapper,
    MotionBlurWrapper,
    PhotometricAugWrapper,
    RandomFrameRepeatingWrapper,
    ResizeCropWrapper,
    YellowLaneAugWrapper,
)
from wrappers.reward_wrappers import LaneFollowingRewardWrapper
from wrappers.respawn_wrapper import maybe_wrap_respawn

from .duckiematrix_env import DuckiematrixDB21JEnv
from .map_interpreter_patch import use_patched_map_interpreter
from .map_utils import discover_maps, map_engine_arg


DEFAULT_MAX_EPISODE_STEPS = 300
DEFAULT_FORWARD_SPEED = 1.0
DEFAULT_MAX_STEER = 1.0
DEFAULT_HEADING_TYPE = "heading"
DEFAULT_FRAME_REPEAT_PROB = 0.0
DEFAULT_MOTION_BLUR_KERNEL_SIZE = 0
DEFAULT_PHOTOMETRIC_AUG_STRENGTH = 0.0
DEFAULT_YELLOW_LANE_AUG_STRENGTH = 0.0
DEFAULT_OBSERVATION_MODE = "rgb"
VALID_OBSERVATION_MODES = ("rgb", "binary_lane")
VALID_RESPAWN_BACKENDS = ("engine", "wrapper", "hybrid")


def normalize_respawn_backend(respawn_backend: str | None) -> str:
    backend = (respawn_backend or "wrapper").strip().lower()
    if backend not in VALID_RESPAWN_BACKENDS:
        choices = ", ".join(VALID_RESPAWN_BACKENDS)
        raise ValueError(f"Unsupported respawn backend '{respawn_backend}'. Expected one of: {choices}")
    return backend


def make_single_env(
    entity_name: str = "map_0/vehicle_0",
    headless: bool = True,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    respawn_mode: str = "random",
    respawn_backend: str = "wrapper",
    respawn_kwargs: dict | None = None,
    reward_kwargs: dict | None = None,
    obs_size: tuple[int, int] = (80, 160),
    crop_top_ratio: float = 0.33,
    forward_speed: float = DEFAULT_FORWARD_SPEED,
    forward_speed_min: float | None = None,
    forward_speed_max: float | None = None,
    max_steer: float = DEFAULT_MAX_STEER,
    heading_type: str = DEFAULT_HEADING_TYPE,
    frame_repeat_prob: float = DEFAULT_FRAME_REPEAT_PROB,
    motion_blur_kernel_size: int = DEFAULT_MOTION_BLUR_KERNEL_SIZE,
    photometric_aug_strength: float = DEFAULT_PHOTOMETRIC_AUG_STRENGTH,
    yellow_lane_aug_strength: float = DEFAULT_YELLOW_LANE_AUG_STRENGTH,
    observation_mode: str = DEFAULT_OBSERVATION_MODE,
    lane_mask_noise_strength: float = DEFAULT_LANE_MASK_NOISE_STRENGTH,
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

    if normalize_respawn_backend(respawn_backend) in ("wrapper", "hybrid"):
        env = maybe_wrap_respawn(
            env,
            respawn_mode=respawn_mode,
            respawn_kwargs=respawn_kwargs,
        )

    reward_kwargs = reward_kwargs or {}
    env = LaneFollowingRewardWrapper(env, **reward_kwargs)

    out_h, out_w = obs_size
    env = ResizeCropWrapper(env, out_h=out_h, out_w=out_w, crop_top_ratio=crop_top_ratio)

    observation_mode = str(observation_mode).strip().lower()
    if observation_mode not in VALID_OBSERVATION_MODES:
        choices = ", ".join(VALID_OBSERVATION_MODES)
        raise ValueError(f"Unsupported observation_mode '{observation_mode}'. Expected one of: {choices}")

    if observation_mode == "rgb" and yellow_lane_aug_strength > 0.0:
        env = YellowLaneAugWrapper(env, strength=yellow_lane_aug_strength)
    if photometric_aug_strength > 0.0:
        env = PhotometricAugWrapper(env, strength=photometric_aug_strength)
    if frame_repeat_prob > 0.0:
        env = RandomFrameRepeatingWrapper(env, repeat_prob=frame_repeat_prob)
    if motion_blur_kernel_size > 1:
        env = MotionBlurWrapper(env, kernel_size=motion_blur_kernel_size)
    if observation_mode == "binary_lane":
        env = BinaryLaneMaskWrapper(env)
        if lane_mask_noise_strength > 0.0:
            env = LaneMaskNoiseWrapper(env, strength=lane_mask_noise_strength)

    env = HeadingToWheelsWrapper(
        env,
        forward_speed=forward_speed,
        forward_speed_min=forward_speed_min,
        forward_speed_max=forward_speed_max,
        max_steer=max_steer,
        heading_type=heading_type,
    )
    if max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
