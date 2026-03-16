import logging
import math

import gymnasium as gym

import lane_utils as lu


VALID_RESPAWN_MODES = ("random", "fixed")
RESPAWN_MODE_ALIASES = {
    "0": "fixed",
    "false": "fixed",
    "no": "fixed",
    "default": "fixed",
    "deterministic": "fixed",
    "engine": "fixed",
    "manual": "fixed",
    "map": "fixed",
    "1": "random",
    "true": "random",
    "yes": "random",
}


logger = logging.getLogger(__name__)


def normalize_respawn_mode(respawn_mode: str | None) -> str:
    mode = (respawn_mode or "random").strip().lower()
    mode = RESPAWN_MODE_ALIASES.get(mode, mode)
    if mode not in VALID_RESPAWN_MODES:
        choices = ", ".join(VALID_RESPAWN_MODES)
        raise ValueError(f"Unsupported respawn mode '{respawn_mode}'. Expected one of: {choices}")
    return mode


def maybe_wrap_respawn(
    env: gym.Env,
    respawn_mode: str = "random",
    respawn_kwargs: dict | None = None,
) -> gym.Env:
    if normalize_respawn_mode(respawn_mode) != "random":
        return env
    return RandomRespawnWrapper(env, **(respawn_kwargs or {}))


class RandomRespawnWrapper(gym.Wrapper):
    """
    Retry reset until a new starting pose satisfies the spawn constraints.

    The wrapper prefers the base environment's random respawn API when it is
    available. If not, it falls back to a normal reset and validates the tile
    and heading after each attempt.
    """

    def __init__(
        self,
        env: gym.Env,
        lateral_jitter: float = 0.02,
        yaw_jitter_deg: float = 0.0,
        avoid_junction: bool = True,
        fallback_bbox=None,
        max_reset_attempts: int | None = None,
        max_spawn_angle_deg: float = 60.0,
    ):
        super().__init__(env)
        self.lateral_jitter = lateral_jitter
        self.yaw_jitter_deg = yaw_jitter_deg
        self.avoid_junction = avoid_junction
        self.fallback_bbox = fallback_bbox
        self.max_reset_attempts = None
        self.max_spawn_angle_deg = max(0.0, float(max_spawn_angle_deg))
        self.last_reset_attempts = 0
        self.last_reset_succeeded = False
        self.last_reset_dot = None
        self.last_reset_angle_deg = None
        self.last_reset_tile = None

    def _base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def _refresh_obs_and_info(self, obs, info):
        base_env = self._base_env()
        try:
            if hasattr(base_env, "_get_pose_blocking"):
                base_env.last_pose = base_env._get_pose_blocking()
            if hasattr(base_env, "_get_rgb_frame_blocking"):
                obs = base_env._get_rgb_frame_blocking()
            if hasattr(base_env, "_get_info"):
                info = base_env._get_info()
        except Exception:
            pass
        return obs, info

    def _try_random_respawn(self, obs, info):
        if not hasattr(self.env, "random_respawn"):
            return obs, info

        try:
            self.env.random_respawn(
                lateral_jitter=self.lateral_jitter,
                yaw_jitter_deg=self.yaw_jitter_deg,
                avoid_junction=self.avoid_junction,
                fallback_bbox=self.fallback_bbox,
            )
            obs, info = self._refresh_obs_and_info(obs, info)
        except Exception:
            pass
        return obs, info

    def _spawn_status(self):
        base_env = self._base_env()
        pose = getattr(base_env, "last_pose", None)
        lp_cal = getattr(base_env, "lp_cal", None)
        if pose is None or lp_cal is None:
            return {
                "valid": True,
                "tile": None,
                "drivable": None,
                "dot": None,
            }

        try:
            pos_lane, yaw = lu.pose_to_lane_frame(pose, lp_cal=lp_cal)
            tile, _, drivable = lu.tile_info(lp_cal, pos_lane)
            dot = None
            angle_deg = None
            if drivable:
                lp = lu.get_lane_pos(lp_cal, pos_lane, float(yaw))
                dot = float(lp.dot_dir)
                angle_deg = abs(float(lp.angle_deg))
            valid = bool(drivable) and (angle_deg is not None) and (float(angle_deg) <= self.max_spawn_angle_deg)
            return {
                "valid": bool(valid),
                "tile": tuple(tile) if tile is not None else None,
                "drivable": bool(drivable),
                "dot": dot,
                "angle_deg": angle_deg,
            }
        except Exception:
            return {
                "valid": False,
                "tile": None,
                "drivable": False,
                "dot": None,
                "angle_deg": None,
            }

    def reset(self, **kwargs):
        obs = None
        info = {}
        attempt = 0
        while True:
            attempt += 1
            obs, info = self.env.reset(**kwargs)
            obs, info = self._try_random_respawn(obs, info)
            status = self._spawn_status()
            self.last_reset_tile = status["tile"]
            self.last_reset_dot = status["dot"]
            self.last_reset_angle_deg = status["angle_deg"]
            if status["valid"]:
                self.last_reset_attempts = attempt
                self.last_reset_succeeded = True
                if isinstance(info, dict):
                    info = dict(info)
                    info["respawn_tile"] = self.last_reset_tile
                    info["respawn_dot"] = self.last_reset_dot
                    info["respawn_angle_deg"] = self.last_reset_angle_deg
                return obs, info
            if attempt % 50 == 0:
                logger.warning(
                    "RandomRespawnWrapper still searching for a drivable aligned spawn: attempts=%d tile=%r dot=%r angle_deg=%r max_angle_deg=%.1f",
                    attempt,
                    self.last_reset_tile,
                    self.last_reset_dot,
                    self.last_reset_angle_deg,
                    self.max_spawn_angle_deg,
                )
