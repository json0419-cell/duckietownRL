import gymnasium as gym
import numpy as np


VALID_HEADING_TYPES = ("heading", "heading_smooth")


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
    Map a single steering input to left and right wheel speeds.

    The input range is -1..1, and the output is clipped by the forward speed
    and maximum steering magnitude.
    """
    def __init__(
        self,
        env,
        forward_speed=1.0,
        max_steer=1.0,
        heading_type="heading",
        forward_speed_min=None,
        forward_speed_max=None,
    ):
        super().__init__(env)
        self.forward_speed = float(forward_speed)
        self.max_steer = float(max_steer)
        self.heading_type = str(heading_type or "heading")
        self.forward_speed_min = None if forward_speed_min is None else float(forward_speed_min)
        self.forward_speed_max = None if forward_speed_max is None else float(forward_speed_max)
        if (self.forward_speed_min is None) != (self.forward_speed_max is None):
            raise ValueError("forward_speed_min and forward_speed_max must be provided together")
        if self.forward_speed_min is not None and self.forward_speed_max < self.forward_speed_min:
            raise ValueError("forward_speed_max must be >= forward_speed_min")
        self.current_forward_speed = self.forward_speed
        if self.heading_type not in VALID_HEADING_TYPES:
            choices = ", ".join(VALID_HEADING_TYPES)
            raise ValueError(f"Unsupported heading_type '{self.heading_type}'. Expected one of: {choices}")
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        if self.forward_speed_min is not None:
            self.current_forward_speed = float(np.random.uniform(self.forward_speed_min, self.forward_speed_max))
        else:
            self.current_forward_speed = self.forward_speed
        return self.env.reset(**kwargs)

    def action(self, act):
        if isinstance(act, tuple):
            act = act[0]
        heading_input = np.asarray(act, dtype=np.float32).reshape(-1)
        if heading_input.size == 0:
            raise ValueError("HeadingToWheelsWrapper received an empty action.")

        heading = float(np.clip(heading_input[0], -1.0, 1.0))
        if self.heading_type == "heading_smooth":
            heading = (heading ** 3) * self.max_steer
        else:
            heading = heading * self.max_steer

        wheels = np.array([1.0 + heading, 1.0 - heading], dtype=np.float32)
        wheels = np.clip(wheels, 0.0, 1.0)
        if self.current_forward_speed != 1.0:
            wheels = np.clip(wheels * self.current_forward_speed, 0.0, 1.0)
        return wheels
