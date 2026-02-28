"""
评估/测试已训练的 heading 动作空间智能体。

用法示例：
  python test_agent.py --model basic_no_random_respawn_single_map/ppo_db21j_final.zip --episodes 5 --headless
"""
import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from reward_wrappers import LaneFollowingRewardWrapper
from observation_wrappers import ResizeCropWrapper
from action_wrappers import HeadingToWheelsWrapper


def make_single_env(headless: bool, reward_kwargs: dict, obs_size=(80, 160), crop_top_ratio=0.33,
                    forward_speed: float = 0.6, max_steer: float = 1.0):
    env = DuckiematrixDB21JEnv(
        entity_name="map_0/vehicle_0",
        out_of_road_penalty=-10.0,
        headless=headless,
        camera_height=480,
        camera_width=640,
    )
    env = LaneFollowingRewardWrapper(env, **reward_kwargs)
    out_h, out_w = obs_size
    env = ResizeCropWrapper(env, out_h=out_h, out_w=out_w, crop_top_ratio=crop_top_ratio)
    env = HeadingToWheelsWrapper(env, forward_speed=forward_speed, max_steer=max_steer)
    return env


def build_vec_env(headless: bool, reward_kwargs: dict, forward_speed: float, max_steer: float):
    def _factory():
        return make_single_env(
            headless=headless,
            reward_kwargs=reward_kwargs,
            obs_size=(80, 160),
            crop_top_ratio=0.33,
            forward_speed=forward_speed,
            max_steer=max_steer,
        )

    venv = DummyVecEnv([_factory])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=3)
    return venv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="path to PPO model zip, e.g. basic_no_random_respawn_single_map/ppo_db21j_final.zip")
    p.add_argument("--episodes", type=int, default=5, help="number of evaluation episodes")
    p.add_argument("--headless", action="store_true", help="run without rendering (recommended for speed)")
    p.add_argument("--forward_speed", type=float, default=1.0, help="heading wrapper forward speed")
    p.add_argument("--max_steer", type=float, default=1.0, help="heading wrapper max steer scale")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    reward_kwargs = {
        "w_forward": 1.0,
        "w_center": 1.0,
        "w_smooth": 0.05,
        "center_sigma": 0.10,
        "speed_clip": 5.0,
        "enforce_right_lane": True,
        "right_lane_min_dot": 0.2,
        "right_lane_min_dist": -0.02,
        "right_lane_curve_margin": 0.05,
        "min_speed_reward": 0.02,
    }

    venv = build_vec_env(
        headless=args.headless,
        reward_kwargs=reward_kwargs,
        forward_speed=args.forward_speed,
        max_steer=args.max_steer,
    )

    model = PPO.load(args.model, env=venv)

    episode_rewards = []
    for ep in range(args.episodes):
        obs = venv.reset()
        done_flag = False
        ep_rew = 0.0
        while not done_flag:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            r = float(np.asarray(reward)[0])
            done_flag = bool(np.asarray(done)[0])
            ep_rew += r
        episode_rewards.append(ep_rew)
        print(f"Episode {ep+1}/{args.episodes}: reward={ep_rew:.3f}")

    mean_rew = np.mean(episode_rewards)
    std_rew = np.std(episode_rewards)
    print(f"Finished {args.episodes} episodes. Mean reward={mean_rew:.3f} ± {std_rew:.3f}")
    venv.close()


if __name__ == "__main__":
    main()
