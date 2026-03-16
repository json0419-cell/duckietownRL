import gymnasium as gym
import numpy as np
from typing import Dict, Tuple

import lane_utils as lu
from lane_utils import get_lane_pos, pose_to_lane_frame


def _find_unwrapped(env):
    e = env
    seen = set()
    while True:
        eid = id(e)
        if eid in seen:
            break
        seen.add(eid)

        if hasattr(e, "env") and getattr(e, "env") is not None:
            e = e.env
            continue
        if hasattr(e, "unwrapped"):
            try:
                e = e.unwrapped
                continue
            except Exception:
                pass
        break
    return e


class LaneFollowingRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        reward_mode: str = "posangle",
        include_velocity_reward: bool = True,
        include_collision_avoidance: bool = True,
        max_lp_dist: float = 0.05,
        max_dev_from_target_angle_deg_narrow: float = 10.0,
        max_dev_from_target_angle_deg_wide: float = 50.0,
        target_angle_deg_at_edge: float = 45.0,
        orientation_scale: float = 0.5,
        velocity_reward_scale: float = 0.25,
        **legacy_kwargs,
    ):
        super().__init__(env)

        self.reward_mode = str(reward_mode)
        self.include_velocity_reward = bool(include_velocity_reward)
        self.include_collision_avoidance = bool(include_collision_avoidance)
        self.max_lp_dist = float(max_lp_dist)
        self.max_dev_from_target_angle_deg_narrow = float(max_dev_from_target_angle_deg_narrow)
        self.max_dev_from_target_angle_deg_wide = float(max_dev_from_target_angle_deg_wide)
        self.target_angle_deg_at_edge = float(target_angle_deg_at_edge)
        self.orientation_scale = float(orientation_scale)
        self.velocity_reward_scale = float(velocity_reward_scale)
        self.legacy_kwargs = dict(legacy_kwargs)

        self.last_pose = None
        self.last_actions = np.array([0.0, 0.0], dtype=np.float32)
        self.last_speed = 0.0
        self.last_lp = None
        self.prev_proximity_penalty = 0.0
        self.orientation_reward = 0.0
        self.velocity_reward = 0.0
        self.collision_avoidance_reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        base = _find_unwrapped(self.env)
        self.last_pose = getattr(base, "last_pose", None)
        self.last_actions = np.array([0.0, 0.0], dtype=np.float32)
        self.last_speed = 0.0
        self.last_lp = None
        self.prev_proximity_penalty = 0.0
        self.orientation_reward = 0.0
        self.velocity_reward = 0.0
        self.collision_avoidance_reward = 0.0
        return obs, info

    def step(self, action) -> Tuple:
        self.last_actions = np.asarray(action, dtype=np.float32).copy()

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        base = _find_unwrapped(self.env)
        pose = getattr(base, "last_pose", None)
        reward, dbg = self._duckietown_rl_reward(base, pose, float(base_reward), bool(terminated))

        info = dict(info)
        info["speed"] = self.last_speed
        info["lp_dist"] = float(self.last_lp.dist) if self.last_lp is not None else None
        info["lp_dot_dir"] = float(self.last_lp.dot_dir) if self.last_lp is not None else None
        info["debug_lane_reward"] = dbg

        return obs, float(reward), terminated, truncated, info

    @staticmethod
    def _leaky_cosine(x: float) -> float:
        slope = 0.05
        if abs(x) < np.pi:
            return float(np.cos(x))
        return float(-1.0 - slope * (abs(x) - np.pi))

    def _calculate_pos_angle_reward(self, lp_dist: float, lp_angle_deg: float) -> Tuple[float, float, float]:
        normed_lp_dist = float(lp_dist) / self.max_lp_dist
        clipped_dist = float(np.clip(normed_lp_dist, -1.0, 1.0))
        target_angle_deg = -clipped_dist * self.target_angle_deg_at_edge
        narrow = 0.5 + 0.5 * self._leaky_cosine(
            np.pi * (target_angle_deg - float(lp_angle_deg)) / self.max_dev_from_target_angle_deg_narrow
        )
        wide = 0.5 + 0.5 * self._leaky_cosine(
            np.pi * (target_angle_deg - float(lp_angle_deg)) / self.max_dev_from_target_angle_deg_wide
        )
        return float(narrow), float(wide), float(target_angle_deg)

    def _wheel_velocity_reward(self, base_env) -> float:
        if not self.include_velocity_reward:
            return 0.0

        wheel_vels = getattr(base_env, "wheelVels", None)
        if wheel_vels is None:
            wheel_vels = self.last_actions

        vel_reward = float(np.max(np.asarray(wheel_vels, dtype=np.float32))) * self.velocity_reward_scale
        if np.isnan(vel_reward):
            vel_reward = 0.0
        return float(vel_reward)

    def _collision_reward(self, base_env) -> Tuple[float, bool]:
        if not self.include_collision_avoidance:
            return 0.0, False

        has_proximity = (
            hasattr(base_env, "proximity_penalty2")
            and hasattr(base_env, "cur_pos")
            and hasattr(base_env, "cur_angle")
        )
        if not has_proximity:
            return 0.0, False

        proximity_penalty = float(base_env.proximity_penalty2(base_env.cur_pos, base_env.cur_angle))
        proximity_reward = -(self.prev_proximity_penalty - proximity_penalty) * 50.0
        if proximity_reward < 0.0:
            proximity_reward = 0.0
        self.prev_proximity_penalty = proximity_penalty
        return float(proximity_reward), True

    def _duckietown_rl_reward(self, base_env, pose: Dict, base_reward: float, terminated: bool) -> Tuple[float, Dict]:
        dbg = {
            "reward_mode": self.reward_mode,
            "terminated_from_env": bool(terminated),
            "speed": None,
            "pose_valid": None,
            "invalid_points": [],
            "first_invalid_point": None,
            "dist_signed": None,
            "dot_dir": None,
            "half": None,
            "min_dot": None,
            "min_dist": None,
            "violated": False,
            "lp_dist": None,
            "lp_dot_dir": None,
            "lp_angle_deg": None,
            "target_angle_deg": None,
            "orientation_reward": 0.0,
            "velocity_reward": 0.0,
            "collision_avoidance_reward": 0.0,
            "collision_reward_available": False,
            "base_reward": float(base_reward),
            "reward": None,
            "reasons": [],
            "legacy_kwargs_ignored": sorted(self.legacy_kwargs.keys()),
        }

        if pose is None or self.last_pose is None:
            dbg["reasons"].append("missing_pose")
            dbg["reward"] = float(base_reward)
            return float(base_reward), dbg

        t = float(pose["header"]["timestamp"])
        last_t = float(self.last_pose["header"]["timestamp"])
        dt = max(t - last_t, 1e-3)

        x = float(pose["position"]["x"])
        y = float(pose["position"]["y"])
        z = float(pose["position"]["z"])
        lx = float(self.last_pose["position"]["x"])
        ly = float(self.last_pose["position"]["y"])
        lz = float(self.last_pose["position"]["z"])
        speed = np.sqrt((x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2) / dt
        self.last_speed = float(speed)
        dbg["speed"] = self.last_speed

        lp_cal = base_env.lp_cal
        pos_lane, yaw_pose = pose_to_lane_frame(pose, lp_cal=lp_cal)
        pose_report = lu.valid_pose_report(lp_cal, pos_lane, float(yaw_pose))
        dbg["pose_valid"] = bool(pose_report["valid"])
        dbg["invalid_points"] = list(pose_report["invalid_points"])
        dbg["first_invalid_point"] = pose_report["first_invalid_point"]

        reward = float(base_reward)
        orientation_reward = 0.0

        if self.reward_mode in ("posangle", "Posangle", "target_orientation"):
            try:
                lp = get_lane_pos(lp_cal, pos_lane, float(yaw_pose))
                self.last_lp = lp
                dbg["lp_dist"] = float(lp.dist)
                dbg["lp_dot_dir"] = float(lp.dot_dir)
                dbg["lp_angle_deg"] = float(lp.angle_deg)
                dbg["dist_signed"] = float(lp.dist)
                dbg["dot_dir"] = float(lp.dot_dir)

                narrow, wide, target_angle_deg = self._calculate_pos_angle_reward(lp.dist, lp.angle_deg)
                dbg["target_angle_deg"] = float(target_angle_deg)

                if self.reward_mode == "target_orientation":
                    orientation_reward = narrow
                else:
                    orientation_reward = self.orientation_scale * (narrow + wide)
            except Exception:
                self.last_lp = None
                orientation_reward = -10.0
                dbg["reasons"].append("not_in_lane")

            reward = float(orientation_reward)
        elif self.reward_mode == "default":
            try:
                lp = get_lane_pos(lp_cal, pos_lane, float(yaw_pose))
                self.last_lp = lp
                dbg["lp_dist"] = float(lp.dist)
                dbg["lp_dot_dir"] = float(lp.dot_dir)
                dbg["lp_angle_deg"] = float(lp.angle_deg)
            except Exception:
                self.last_lp = None
                dbg["reasons"].append("not_in_lane")
        else:
            dbg["reasons"].append(f"unknown_reward_mode:{self.reward_mode}")

        velocity_reward = self._wheel_velocity_reward(base_env)
        collision_reward, collision_available = self._collision_reward(base_env)

        self.orientation_reward = float(orientation_reward)
        self.velocity_reward = float(velocity_reward)
        self.collision_avoidance_reward = float(collision_reward)

        dbg["orientation_reward"] = self.orientation_reward
        dbg["velocity_reward"] = self.velocity_reward
        dbg["collision_avoidance_reward"] = self.collision_avoidance_reward
        dbg["collision_reward_available"] = bool(collision_available)

        reward = float(reward + velocity_reward + collision_reward)
        dbg["reward"] = float(reward)

        self.last_pose = pose
        return float(reward), dbg
