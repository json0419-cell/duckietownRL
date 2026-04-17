from collections import deque

import gymnasium as gym
import numpy as np


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

