# train_ppo_db21j_full.py
import os
import argparse
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# your modules (ensure these files exist as discussed)
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from reward_wrappers import LaneFollowingRewardWrapper
from observation_wrappers import ResizeCropWrapper
from action_wrappers import HeadingToWheelsWrapper
from respawn_wrapper import RandomRespawnWrapper

def make_single_env(
    entity_name: str = "map_0/vehicle_0",
    headless: bool = False,
    max_episode_steps: int = 500,
    respawn_kwargs: dict = None,
    reward_kwargs: dict = None,
    obs_size: tuple = (80, 160),
    crop_top_ratio: float = 0.33,
    forward_speed: float = 0.6,
    max_steer: float = 1.0,
):
    """
    顺序（从里到外）：
      1) 基础 env (DuckiematrixDB21JLaneFollowEnv)
      2) RandomRespawnWrapper (在 reset 时设置随机初始位姿)
      3) LaneFollowingRewardWrapper (替换 reward)
      4) ResizeCropWrapper (只改观测)
      5) HeadingToWheelsWrapper (单转向动作，速度固定)
    注意：VecTransposeImage/VecFrameStack 在外面处理（针对 VecEnv）
    """
    # 1) 基础 env
    env = DuckiematrixDB21JEnv(
        entity_name=entity_name,
        out_of_road_penalty=-10.0,
        headless=headless,
        camera_height=480,
        camera_width=640,
    )

    # 2) 随机重生（如可用）
    if respawn_kwargs is None:
        respawn_kwargs = {}
    env = RandomRespawnWrapper(env, **respawn_kwargs)

    # 3) Reward wrapper (替换 reward)
    reward_kwargs = reward_kwargs or {}
    env = LaneFollowingRewardWrapper(env, **reward_kwargs)

    # 4) Observation preprocessing (crop + resize)
    out_h, out_w = obs_size
    env = ResizeCropWrapper(env, out_h=out_h, out_w=out_w, crop_top_ratio=crop_top_ratio)

    # 5) Action reparam: agent 输出 [heading]，速度固定
    env = HeadingToWheelsWrapper(env, forward_speed=forward_speed, max_steer=max_steer)

    # 6) Optional TimeLimit wrapper around the whole stack
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000, help="total timesteps to train")
    p.add_argument("--logdir", type=str, default="./runs_db21j_ppo", help="logging/checkpoint dir")
    p.add_argument("--entity-name", type=str, default="map_0/vehicle_0", help="target vehicle entity name")
    p.add_argument("--headless", action="store_true", help="run headless (default False)")
    p.add_argument("--num_envs", type=int, default=1, help="parallel envs (use 1 for DummyVecEnv)")
    p.add_argument("--save-freq", type=int, default=200_000, help="checkpoint frequency (timesteps)")
    p.add_argument("--save-name", type=str, default="ppo_db21j_final", help="final model filename")
    p.add_argument("--load-model", type=str, default=None, help="path to existing PPO model .zip to resume training")
    p.add_argument("--device", type=str, default="auto", help="torch device: auto/cpu/cuda")
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
        "enforce_right_lane": True,
        # 必须顺行（朝向与车道切线夹角小于约78°），否则视为逆行
        "right_lane_min_dot": 0.2,
        # 车体中心需在车道中心线右侧，但允许 2cm 容差，避免微小浮点误差导致奖励归零
        "right_lane_min_dist": -0.02,
        # 曲线额外放宽 5cm，避免切线抖动导致误罚
        "right_lane_curve_margin": 0.05,
        # 低于此速度（m/s）不给正奖励，防止停车刷分
        "min_speed_reward": 0.02,
    }

    # create single env factory (for DummyVecEnv)
    def _env_factory():
        return make_single_env(
            entity_name=args.entity_name,
            headless=args.headless,
            max_episode_steps=500,
            respawn_kwargs=respawn_kwargs,
            reward_kwargs=reward_kwargs,
            obs_size=(80, 160),
            crop_top_ratio=0.33,
            forward_speed=1.0,
            max_steer=1.0,
        )

    # Vectorized env (single proc). 如果你能并行运行多个模拟，改用 SubprocVecEnv
    venv = DummyVecEnv([_env_factory for _ in range(max(1, args.num_envs))])

    # SB3 的 CNN policy 要求 CHW 格式，所以先转置
    venv = VecTransposeImage(venv)

    # stack frames: 给策略 temporal 信息 (n_stack=3 为 Duckietown-RL 的常见设置)
    venv = VecFrameStack(venv, n_stack=3)

    # checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=logdir,
        name_prefix="ppo_db21j",
    )

    # PPO agent
    if args.load_model:
        print(f"[INFO] Resuming training from: {args.load_model}")
        model = PPO.load(args.load_model, env=venv, device=args.device)
    else:
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
            device=args.device,
        )
    # 记录到 stdout / CSV / TensorBoard，便于后续出图
    new_logger = configure(logdir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # start training
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=(args.load_model is None),
    )

    # final save
    final_path = os.path.join(logdir, args.save_name)
    model.save(final_path)
    saved_file = final_path if final_path.endswith(".zip") else f"{final_path}.zip"
    print("Training finished. Model saved to:", saved_file)


if __name__ == "__main__":
    main()
