import gymnasium as gym
import numpy as np
from typing import Tuple, Dict

from lane_utils import pose_to_lane_frame, is_in_lane, get_lane_pos

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
    """

    def __init__(
        self,
        env: gym.Env,
        w_forward: float = 1.0,
        w_center: float = 1.0,
        w_smooth: float = 0.05,
        center_sigma: float = 0.10,
        speed_clip: float = 5.0,
    ):
        super().__init__(env)

        self.w_forward = w_forward
        self.w_center = w_center
        self.w_smooth = w_smooth
        self.center_sigma = center_sigma
        self.speed_clip = speed_clip

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
        reward, terminated2 = self._lane_reward(pose)

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

        return obs, reward, terminated, truncated, info

    def _lane_reward(self, pose: Dict) -> Tuple[float, bool]:
        assert self.last_pose is not None

        terminated = False
        last_pose = self.last_pose

        t = float(pose["header"]["timestamp"])
        last_t = float(last_pose["header"]["timestamp"])
        dt = max(t - last_t, 1e-3)

        x, y, z = pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]
        lx, ly, lz = last_pose["position"]["x"], last_pose["position"]["y"], last_pose["position"]["z"]

        dx, dy, dz = x - lx, y - ly, z - lz
        speed = np.sqrt(dx*dx + dy*dy + dz*dz) / dt
        speed = float(np.clip(speed, 0.0, self.speed_clip))
        self.last_speed = speed

        pos_lane, yaw = pose_to_lane_frame(pose, lp_cal=self.env.lp_cal)

        if not is_in_lane(self.env.lp_cal, pos_lane, float(yaw)):
            reward = float(self.env.out_of_road_penalty)
            terminated = True
            self.last_lp = None
        else:
            try:
                lp = get_lane_pos(
                    self.env.lp_cal,
                    pos_lane,
                    float(yaw),
                )
                self.last_lp = lp

                r_forward = speed * float(lp.dot_dir)
                d = float(lp.dist)
                r_center = np.exp(-(d * d) / (self.center_sigma ** 2))

                wl, wr = self.last_actions
                r_smooth = -abs(wl - wr)

                reward = (
                    self.w_forward * r_forward
                    + self.w_center * r_center
                    + self.w_smooth * r_smooth
                )

            except Exception:
                reward = float(self.env.out_of_road_penalty)
                terminated = True
                self.last_lp = None

        self.last_pose = pose
        return float(reward), terminated
