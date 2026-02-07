import gymnasium as gym
import numpy as np
import cv2


class ResizeCropWrapper(gym.ObservationWrapper):
    def __init__(self, env, out_h=80, out_w=160, crop_top_ratio=0.33):
        super().__init__(env)
        self.out_h = out_h
        self.out_w = out_w
        self.crop_top_ratio = crop_top_ratio

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(out_h, out_w, 3), dtype=np.uint8
        )

    def observation(self, obs):
        H = obs.shape[0]
        top = int(H * self.crop_top_ratio)
        img = obs[top:, :, :]
        img = cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return img
