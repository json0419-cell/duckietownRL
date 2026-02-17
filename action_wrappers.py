import gymnasium as gym
import numpy as np


class ThrottleSteerToWheelsWrapper(gym.ActionWrapper):
    def __init__(self, env, min_throttle=0.2):
        super().__init__(env)
        self.min_throttle = min_throttle

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

    def action(self, act):
        throttle, steer = np.clip(act, -1.0, 1.0)

        t01 = (throttle + 1.0) * 0.5
        t = self.min_throttle + (1.0 - self.min_throttle) * t01

        wl = t * (1.0 + steer) * 0.5
        wr = t * (1.0 - steer) * 0.5

        return np.array([wl, wr], dtype=np.float32)


class HeadingToWheelsWrapper(gym.ActionWrapper):
    """
    单输入：期望朝向偏转（左负右正，-1..1），固定前进速度。
    适用于“heading”模式：策略只管转向，速度恒定。
    """
    def __init__(self, env, forward_speed=0.5, max_steer=1.0):
        super().__init__(env)
        self.forward_speed = float(forward_speed)
        self.max_steer = float(max_steer)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, act):
        steer = float(np.clip(act[0], -1.0, 1.0)) * self.max_steer
        t = self.forward_speed
        wl = t * (1.0 + steer) * 0.5
        wr = t * (1.0 - steer) * 0.5
        return np.array([wl, wr], dtype=np.float32)
