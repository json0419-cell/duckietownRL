"""
Evaluate a trained single-input steering agent.

Example:
  python test_agent.py --model basic_no_random_respawn_single_map/ppo_db21j_final.zip --episodes 5 --headless
"""
import argparse
import os
import sys
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from duckiematrix_env import DuckiematrixDB21JEnv
from reward_wrappers import LaneFollowingRewardWrapper
from observation_wrappers import ResizeCropWrapper
from action_wrappers import HeadingToWheelsWrapper
from respawn_wrapper import VALID_RESPAWN_MODES, maybe_wrap_respawn
from map_interpreter_patch import use_patched_map_interpreter


def make_single_env(
    headless: bool,
    reward_kwargs: dict,
    respawn_mode: str,
    obs_size=(80, 160),
    crop_top_ratio=0.33,
    forward_speed: float = 1.0,
    max_steer: float = 1.0,
):
    env = DuckiematrixDB21JEnv(
        entity_name="map_0/vehicle_0",
        out_of_road_penalty=-10.0,
        headless=headless,
        camera_height=480,
        camera_width=640,
    )
    use_patched_map_interpreter(env)
    env = maybe_wrap_respawn(env, respawn_mode=respawn_mode)
    env = LaneFollowingRewardWrapper(env, **reward_kwargs)
    out_h, out_w = obs_size
    env = ResizeCropWrapper(env, out_h=out_h, out_w=out_w, crop_top_ratio=crop_top_ratio)
    env = HeadingToWheelsWrapper(env, forward_speed=forward_speed, max_steer=max_steer)
    return env


def build_vec_env(
    headless: bool,
    reward_kwargs: dict,
    respawn_mode: str,
    forward_speed: float,
    max_steer: float,
):
    def _factory():
        return make_single_env(
            headless=headless,
            reward_kwargs=reward_kwargs,
            respawn_mode=respawn_mode,
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
    p.add_argument(
        "--respawn-mode",
        type=str,
        default="fixed",
        choices=VALID_RESPAWN_MODES,
        help="respawn mode for evaluation episodes",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    reward_kwargs = {
        "reward_mode": "posangle",
        "include_velocity_reward": True,
    }

    venv = build_vec_env(
        headless=args.headless,
        reward_kwargs=reward_kwargs,
        respawn_mode=args.respawn_mode,
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
