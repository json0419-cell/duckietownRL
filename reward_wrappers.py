import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Optional

import lane_utils as lu
from lane_utils import pose_to_lane_frame, is_in_lane, get_lane_pos, yaw_from_displacement

def _find_unwrapped(env):
    e = env
    # unwrap through common attributes
    seen = set()
    while True:
        # 防止死循环
        eid = id(e)
        if eid in seen:
            break
        seen.add(eid)

        if hasattr(e, "env") and getattr(e, "env") is not None:
            e = e.env
            continue
        # some wrappers expose .unwrapped
        if hasattr(e, "unwrapped"):
            try:
                e = e.unwrapped
                continue
            except Exception:
                pass
        break
    return e

class LaneFollowingRewardWrapper(gym.Wrapper):
    """
    Reward wrapper for Duckiematrix DB21J:
    - uses lane_utils.get_lane_pos
    - replaces env reward with lane-following reward
    - optional right-lane enforcement (right-hand traffic)
    """

    def __init__(
        self,
        env: gym.Env,
        w_forward: float = 1.0,
        w_center: float = 1.0,
        w_smooth: float = 0.05,
        center_sigma: float = 0.10,
        speed_clip: float = 5.0,
        use_vel_yaw: bool = True,
        vel_yaw_min_dist: float = 1e-4,
        enforce_right_lane: bool = False,
        right_lane_min_dot: float = 0.0,
        right_lane_margin: float = 0.0,
        right_lane_min_dist: Optional[float] = None,
        right_lane_penalty: Optional[float] = None,
        min_speed_reward: float = 0.0,
        right_lane_curve_margin: float = 0.05,
    ):
        super().__init__(env)

        self.w_forward = w_forward
        self.w_center = w_center
        self.w_smooth = w_smooth
        self.center_sigma = center_sigma
        self.speed_clip = speed_clip
        self.use_vel_yaw = use_vel_yaw
        self.vel_yaw_min_dist = vel_yaw_min_dist
        self.enforce_right_lane = enforce_right_lane
        self.right_lane_min_dot = right_lane_min_dot
        self.right_lane_margin = right_lane_margin
        self.right_lane_min_dist = right_lane_min_dist
        self.right_lane_penalty = right_lane_penalty
        self.min_speed_reward = min_speed_reward
        self.right_lane_curve_margin = right_lane_curve_margin

        self.last_pose = None
        self.last_actions = np.array([0.0, 0.0], dtype=np.float32)
        self.last_speed = 0.0
        self.last_lp = None

    def reset(self, **kwargs):
        # 调用底层 reset（可能会被 RandomRespawnWrapper 覆盖 pose）
        obs, info = self.env.reset(**kwargs)

        # 稳健读取最底层 env 的 last_pose
        base = _find_unwrapped(self.env)
        self.last_pose = getattr(base, "last_pose", None)

        # 其它初始化
        self.last_actions = np.array([0.0, 0.0], dtype=np.float32)
        self.last_speed = 0.0
        self.last_lp = None

        return obs, info

    def step(self, action) -> Tuple:
        self.last_actions = np.asarray(action, dtype=np.float32).copy()

        obs, _, terminated, truncated, info = self.env.step(action)

        pose = self.env.last_pose
        reward, terminated2, dbg = self._lane_reward(pose)

        terminated = terminated or terminated2

        # 把 debug 信息塞进 info（不影响 env）
        info = dict(info)
        info["speed"] = self.last_speed
        if self.last_lp is not None:
            info["lp_dist"] = float(self.last_lp.dist)
            info["lp_dot_dir"] = float(self.last_lp.dot_dir)
        else:
            info["lp_dist"] = None
            info["lp_dot_dir"] = None
        info["debug_lane_reward"] = dbg

        return obs, reward, terminated, truncated, info

    def _lane_reward(self, pose: Dict) -> Tuple[float, bool, Dict]:
        assert self.last_pose is not None

        terminated = False
        last_pose = self.last_pose
        dbg = {
            "speed": None,
            "yaw_use": None,
            "yaw_dir": None,
            "dist_signed": None,
            "dist_abs": None,
            "half": None,
            "dot_dir": None,
            "min_dot": float(self.right_lane_min_dot),
            "min_dist": self.right_lane_min_dist,
            "min_dist_eff": None,
            "violated": False,
            "reasons": [],
            "r_forward": None,
            "r_center": None,
            "r_smooth": None,
            "reward": None,
            "tile_kind": None,
        }

        t = float(pose["header"]["timestamp"])
        last_t = float(last_pose["header"]["timestamp"])
        dt = max(t - last_t, 1e-3)

        x, y, z = pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]
        lx, ly, lz = last_pose["position"]["x"], last_pose["position"]["y"], last_pose["position"]["z"]

        dx, dy, dz = x - lx, y - ly, z - lz
        speed = np.sqrt(dx*dx + dy*dy + dz*dz) / dt
        speed = float(np.clip(speed, 0.0, self.speed_clip))
        self.last_speed = speed
        dbg["speed"] = speed

        pos_lane, yaw_pose = pose_to_lane_frame(pose, lp_cal=self.env.lp_cal)
        yaw_use = float(yaw_pose)
        dbg["yaw_use"] = yaw_use
        yaw_vel = None
        if self.use_vel_yaw:
            last_pos_lane, _ = pose_to_lane_frame(last_pose, lp_cal=self.env.lp_cal)
            yaw_vel = yaw_from_displacement(
                pos_lane,
                last_pos_lane,
                min_dist=self.vel_yaw_min_dist,
            )
            if lu.LANE_POS_METHOD == "heading" and yaw_vel is not None:
                yaw_use = yaw_vel
        dbg["yaw_dir"] = yaw_use

        if self.enforce_right_lane:
            try:
                # 侧向距离用保符号版本，方向余弦用对齐版本
                lp_signed = lu.get_lane_pos_by_distance(
                    self.env.lp_cal, pos_lane, float(yaw_use), keep_sign=True
                )
                lp_dir = lu.get_lane_pos_by_distance(
                    self.env.lp_cal, pos_lane, float(yaw_use), keep_sign=False
                )
                dist_val = float(lp_signed.dist)
                dbg["dist_signed"] = dist_val
                dbg["dist_abs"] = abs(dist_val)
                dbg["dot_dir"] = float(lp_dir.dot_dir)

                half = lu.lane_threshold(self.env.lp_cal, pos=pos_lane)
                half = max(0.0, float(half) - float(self.right_lane_margin))
                dbg["half"] = half

                # 曲线处放宽 min_dist：往外侧行驶不罚，避免切线符号抖动
                effective_min_dist = self.right_lane_min_dist
                try:
                    i, j = self.env.lp_cal.get_grid_coords(pos_lane)
                    tile = self.env.lp_cal.map_interpreter.get_tile(i, j)
                    tile_kind = tile.get("kind") if tile else None
                    dbg["tile_kind"] = tile_kind
                    if effective_min_dist is not None and tile_kind == "curve":
                        effective_min_dist -= float(self.right_lane_curve_margin)
                except Exception:
                    pass
                dbg["min_dist_eff"] = effective_min_dist

                if (
                    abs(dist_val) > half
                    or float(lp_dir.dot_dir) < float(self.right_lane_min_dot)
                    or (
                        effective_min_dist is not None
                        and dist_val < float(effective_min_dist)
                    )
                ):
                    dbg["violated"] = True
                    dbg["reasons"].append("right_lane")
                    penalty = self.right_lane_penalty
                    if penalty is None:
                        penalty = 0.0
                    self.last_lp = lp_dir
                    self.last_pose = pose
                    dbg["reward"] = float(penalty)
                    return float(penalty), False, dbg
            except Exception as e:
                dbg["reasons"].append(f"right_lane_error:{e}")

        if not is_in_lane(self.env.lp_cal, pos_lane, float(yaw_use)):
            reward = float(self.env.out_of_road_penalty)
            terminated = True
            self.last_lp = None
            dbg["reasons"].append("out_of_lane")
            dbg["reward"] = float(reward)
            return float(reward), terminated, dbg
        else:
            try:
                lp = get_lane_pos(
                    self.env.lp_cal,
                    pos_lane,
                    float(yaw_use),
                )
                self.last_lp = lp

                # 静止时不给正向奖励（但仍可被罚）
                if speed < float(self.min_speed_reward):
                    reward = 0.0
                    self.last_pose = pose
                    dbg["reasons"].append("min_speed")
                    dbg["reward"] = float(reward)
                    return float(reward), terminated, dbg

                r_forward = speed * float(lp.dot_dir)
                d = float(lp.dist)
                r_center = np.exp(-(d * d) / (self.center_sigma ** 2))

                wl, wr = self.last_actions
                r_smooth = -abs(wl - wr)

                dbg["r_forward"] = r_forward
                dbg["r_center"] = r_center
                dbg["r_smooth"] = r_smooth

                reward = (
                    self.w_forward * r_forward
                    + self.w_center * r_center
                    + self.w_smooth * r_smooth
                )

            except Exception:
                reward = float(self.env.out_of_road_penalty)
                terminated = True
                self.last_lp = None
                dbg["reasons"].append("lp_error")

        self.last_pose = pose
        dbg["reward"] = float(reward)
        return float(reward), terminated, dbg
