import time
import numpy as np
import pygame

import lane_utils as lu
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from reward_wrappers import LaneFollowingRewardWrapper
from action_wrappers import HeadingToWheelsWrapper
from lane_utils import (
    pose_to_lane_frame,
    is_in_lane,
    tile_info,
    lane_threshold,
    LANE_CHECK_MODE,
    LANE_POS_METHOD,
    get_lane_pos,
    get_lane_pos_by_distance,
    yaw_from_displacement,
)

SHOW_POSE = True

# 与训练保持一致的奖励配置
LANE_REWARD_KWARGS = {
    "w_forward": 1.0,
    "w_center": 1.0,
    "w_smooth": 0.05,
    "center_sigma": 0.10,
    "speed_clip": 5.0,
    "enforce_right_lane": True,
    "right_lane_min_dot": 0.2,
    # 与 lane_check.py 保持一致：右侧行驶但允许 2cm 容差
    "right_lane_min_dist": -0.02,
    # 曲线额外放宽 5cm，避免切线抖动导致误罚
    "right_lane_curve_margin": 0.05,
    "min_speed_reward": 0.0,
}


def keys_to_heading(keys, turn=0.6):
    """
    只输出转向信号（-1..1），速度由 HeadingToWheelsWrapper 固定。
    """
    heading = 0.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        heading -= turn
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        heading += turn
    return np.array([float(np.clip(heading, -1.0, 1.0))], dtype=np.float32)


def _fmt_pose(pose, yaw):
    x = float(pose["position"]["x"])
    y = float(pose["position"]["y"])
    z = float(pose["position"]["z"])
    return f"pose=({x:.3f},{y:.3f},{z:.3f}) yaw={yaw:.3f}"


def main():
    base_env = DuckiematrixDB21JEnv(
        entity_name="map_0/vehicle_0",
        out_of_road_penalty=-10.0,
        headless=False,
        camera_height=480,
        camera_width=640,
    )
    # 先包奖励，再包 heading action
    env = LaneFollowingRewardWrapper(base_env, **LANE_REWARD_KWARGS)
    env = HeadingToWheelsWrapper(env, forward_speed=1.0, max_steer=1.0)
    env.reset()
    if SHOW_POSE and base_env.last_pose is not None:
        pos0, yaw0 = pose_to_lane_frame(base_env.last_pose, lp_cal=base_env.lp_cal)
        print(f"Initial {_fmt_pose(base_env.last_pose, yaw0)}")

    pygame.init()
    pygame.display.set_mode((420, 120))
    pygame.display.set_caption(f"Lane test (heading): mode={LANE_CHECK_MODE}")
    clock = pygame.time.Clock()

    print("\nControls: A/D or ←/→ steer, R reset, ESC quit\n")
    print(f"Lane rule: mode={LANE_CHECK_MODE} -> lane_pos={LANE_POS_METHOD}\n")

    step_i = 0
    last_print = 0.0
    last_pos_lane = None
    total_reward = 0.0

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
                break

            if keys[pygame.K_r]:
                env.reset()
                step_i = 0
                last_pos_lane = None
                total_reward = 0.0
                time.sleep(0.15)
                continue

            action = keys_to_heading(keys)
            _, reward, terminated, truncated, info = env.step(action)
            dbg = info.get("debug_lane_reward") if isinstance(info, dict) else None
            total_reward += float(reward)

            pose = base_env.last_pose
            pos, yaw = pose_to_lane_frame(pose, lp_cal=base_env.lp_cal)
            yaw_use = yaw
            if LANE_POS_METHOD == "heading" and last_pos_lane is not None:
                yaw_vel = yaw_from_displacement(pos, last_pos_lane)
                if yaw_vel is not None:
                    yaw_use = yaw_vel
            dir_dot = None
            dir_ok = None
            if last_pos_lane is not None:
                yaw_vel = yaw_from_displacement(pos, last_pos_lane)
                yaw_dir = yaw_vel if yaw_vel is not None else yaw_use
                try:
                    lp_dir = get_lane_pos_by_distance(
                        base_env.lp_cal,
                        pos,
                        float(yaw_dir),
                        keep_sign=True,
                    )
                    dir_dot = float(lp_dir.dot_dir)
                    dir_ok = dir_dot >= 0.0
                except Exception:
                    pass

            (ti, tj), kind, drivable = tile_info(base_env.lp_cal, pos)
            kind_s = kind if kind is not None else "None"

            now = time.time()
            if now - last_print > 0.10:
                last_print = now

                dir_info = f"  dir_dot={dir_dot: .3f}  dir_ok={dir_ok}" if dir_dot is not None else ""

                lp = None
                lp_err = None
                try:
                    lp = get_lane_pos(base_env.lp_cal, pos, yaw_use)
                except Exception as e:
                    lp_err = e

                if lp is not None:
                    half = lane_threshold(base_env.lp_cal, pos=pos)
                    lp_info = f"dist={float(lp.dist): .3f}  half={half: .3f}  dot_dir={float(lp.dot_dir): .3f}"
                else:
                    lp_info = f"lp_error={type(lp_err).__name__}"

                dbg_info = ""
                if dbg is not None:
                    reasons = ",".join(dbg.get("reasons", []))
                    dbg_info = (
                        f"  DBG(speed={dbg.get('speed')!r}, dist={dbg.get('dist_signed')!r}, "
                        f"half={dbg.get('half')!r}, dot={dbg.get('dot_dir')!r}, "
                        f"min_dot={dbg.get('min_dot')!r}, min_dist={dbg.get('min_dist')!r}, "
                        f"violated={dbg.get('violated')!r}, reasons={reasons})"
                    )

                pose_info = _fmt_pose(pose, yaw_use) + "  " if SHOW_POSE else ""
                status = "IN_LANE" if is_in_lane(base_env.lp_cal, pos, yaw_use) else "NOT_IN_LANE"
                print(
                    f"[{step_i:05d}] {pose_info}{status}  tile={ti},{tj} {kind_s} drivable={drivable}  "
                    f"{lp_info}{dir_info}  reward={reward: .3f}  total={total_reward: .3f}{dbg_info}"
                )

            step_i += 1
            last_pos_lane = pos.copy()

            if terminated or truncated:
                print("Episode ended. Press R to reset.")
                time.sleep(0.2)

            clock.tick(30)

    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
