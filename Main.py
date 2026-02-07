# train_ppo_db21j_full.py
import os
import argparse
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# your modules (ensure these files exist as discussed)
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from reward_wrappers import LaneFollowingRewardWrapper
from observation_wrappers import ResizeCropWrapper
from action_wrappers import ThrottleSteerToWheelsWrapper

def make_single_env(
    headless: bool = False,
    max_episode_steps: int = 500,
    respawn_kwargs: dict = None,
    reward_kwargs: dict = None,
    obs_size: tuple = (80, 160),
    crop_top_ratio: float = 0.33,
    min_throttle: float = 0.2,
):
    """
    顺序（从里到外）：
      1) 基础 env (DuckiematrixDB21JLaneFollowEnv)
      2) RandomRespawnWrapper (在 reset 时设置随机初始位姿)
      3) LaneFollowingRewardWrapper (替换 reward)
      4) ResizeCropWrapper (只改观测)
      5) ThrottleSteerToWheelsWrapper (只改 action)
    注意：VecTransposeImage/VecFrameStack 在外面处理（针对 VecEnv）
    """
    # 1) 基础 env
    env = DuckiematrixDB21JEnv(
        entity_name="map_0/vehicle_0",
        out_of_road_penalty=-10.0,
        headless=headless,
        camera_height=480,
        camera_width=640,
    )

    # 3) Reward wrapper (替换 reward)
    reward_kwargs = reward_kwargs or {}
    env = LaneFollowingRewardWrapper(env, **reward_kwargs)

    # 4) Observation preprocessing (crop + resize)
    out_h, out_w = obs_size
    env = ResizeCropWrapper(env, out_h=out_h, out_w=out_w, crop_top_ratio=crop_top_ratio)

    # 5) Action reparam: agent 输出 [throttle, steer]
    env = ThrottleSteerToWheelsWrapper(env, min_throttle=min_throttle)

    # 6) Optional TimeLimit wrapper around the whole stack
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000, help="total timesteps to train")
    p.add_argument("--logdir", type=str, default="./runs_db21j_ppo", help="logging/checkpoint dir")
    p.add_argument("--headless", action="store_true", help="run headless (default False)")
    p.add_argument("--num_envs", type=int, default=1, help="parallel envs (use 1 for DummyVecEnv)")
    return p.parse_args()


def main():
    args = parse_args()
    total_timesteps = args.timesteps
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)

    # wrappers hyperparams (可按需调整)
    respawn_kwargs = {
        "lateral_jitter": 0.02,    # 2 cm
        "yaw_jitter_deg": 8.0,
        "fallback_bbox": None,
        "avoid_junction": True,
    }
    reward_kwargs = {
        "w_forward": 1.0,
        "w_center": 1.0,
        "w_smooth": 0.05,
        "center_sigma": 0.10,
        "speed_clip": 5.0,
    }

    # create single env factory (for DummyVecEnv)
    def _env_factory():
        return make_single_env(
            headless=args.headless,
            max_episode_steps=500,
            respawn_kwargs=respawn_kwargs,
            reward_kwargs=reward_kwargs,
            obs_size=(80, 160),
            crop_top_ratio=0.33,
            min_throttle=0.2,
        )

    # Vectorized env (single proc). 如果你能并行运行多个模拟，改用 SubprocVecEnv
    venv = DummyVecEnv([_env_factory for _ in range(max(1, args.num_envs))])

    # SB3 的 CNN policy 要求 CHW 格式，所以先转置
    venv = VecTransposeImage(venv)

    # stack frames: 给策略 temporal 信息 (n_stack=3 为 Duckietown-RL 的常见设置)
    venv = VecFrameStack(venv, n_stack=3)

    # checkpoint callback
    checkpoint_cb = CheckpointCallback(save_freq=50_000, save_path=logdir, name_prefix="ppo_db21j")

    # PPO agent
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        verbose=1,
        tensorboard_log=logdir,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    # start training
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)

    # final save
    model.save(os.path.join(logdir, "ppo_db21j_final"))
    print("Training finished. Model saved to:", os.path.join(logdir, "ppo_db21j_final"))


if __name__ == "__main__":
    main()
