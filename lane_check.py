import time
import numpy as np
import pygame

from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from lane_utils import (
    pose_to_lane_frame,
    is_in_lane,
    tile_info,
    lane_threshold,
    LANE_CHECK_MODE,
    LANE_POS_METHOD,
    get_lane_pos,
)

SHOW_POSE = True


def keys_to_action(keys, base=0.6, turn=0.35, clip=1.0):
    forward = keys[pygame.K_w] or keys[pygame.K_UP]
    back = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    brake = keys[pygame.K_SPACE]

    wl = 0.0
    wr = 0.0
    if forward:
        wl = base
        wr = base
    elif back:
        wl = -base
        wr = -base

    if left:
        wl -= turn
        wr += turn
    if right:
        wl += turn
        wr -= turn

    if brake:
        wl = 0.0
        wr = 0.0

    wl = float(np.clip(wl, -clip, clip))
    wr = float(np.clip(wr, -clip, clip))
    return np.array([wl, wr], dtype=np.float32)

def _fmt_pose(pose, yaw):
    x = float(pose["position"]["x"])
    y = float(pose["position"]["y"])
    z = float(pose["position"]["z"])
    return f"pose=({x:.3f},{y:.3f},{z:.3f}) yaw={yaw:.3f}"

def main():
    env = DuckiematrixDB21JEnv(
        entity_name="map_0/vehicle_0",
        out_of_road_penalty=-10.0,
        headless=False,
        camera_height=480,
        camera_width=640,
    )
    env.reset()
    if SHOW_POSE and env.last_pose is not None:
        pos0, yaw0 = pose_to_lane_frame(env.last_pose, lp_cal=env.lp_cal)
        print(f"Initial {_fmt_pose(env.last_pose, yaw0)}")

    pygame.init()
    pygame.display.set_mode((420, 120))
    pygame.display.set_caption(f"Lane test: mode={LANE_CHECK_MODE}")
    clock = pygame.time.Clock()

    print("\nControls: WASD/Arrows, SPACE brake, R reset, ESC quit\n")
    print(f"Lane rule: mode={LANE_CHECK_MODE} -> lane_pos={LANE_POS_METHOD}\n")

    step_i = 0
    last_print = 0.0

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
                time.sleep(0.15)
                continue

            action = keys_to_action(keys)
            _, reward, terminated, truncated, _ = env.step(action)

            pose = env.last_pose
            pos, yaw = pose_to_lane_frame(pose, lp_cal=env.lp_cal)

            (ti, tj), kind, drivable = tile_info(env.lp_cal, pos)
            kind_s = kind if kind is not None else "None"

            now = time.time()
            if now - last_print > 0.10:
                last_print = now

                lp = None
                lp_err = None
                try:
                    lp = get_lane_pos(env.lp_cal, pos, yaw)
                except Exception as e:
                    lp_err = e

                if lp is not None:
                    if LANE_CHECK_MODE == "center":
                        half = lane_threshold(env.lp_cal, pos=pos)
                        lp_info = f"dist={float(lp.dist): .3f}  half={half: .3f}  dot_dir={float(lp.dot_dir): .3f}"
                    else:
                        lp_info = f"dist={float(lp.dist): .3f}  dot_dir={float(lp.dot_dir): .3f}"
                else:
                    lp_info = f"lp_error={type(lp_err).__name__}"

                if not is_in_lane(env.lp_cal, pos, yaw):
                    pose_info = _fmt_pose(pose, yaw) + "  " if SHOW_POSE else ""
                    print(
                        f"[{step_i:05d}] {pose_info}NOT_IN_LANE (tile={ti},{tj} {kind_s} drivable={drivable})  "
                        f"{lp_info}  reward={reward: .3f}"
                    )
                else:
                    pose_info = _fmt_pose(pose, yaw) + "  " if SHOW_POSE else ""
                    print(
                        f"[{step_i:05d}] {pose_info}IN_LANE  tile={ti},{tj} {kind_s} drivable={drivable}  "
                        f"{lp_info}  reward={reward: .3f}"
                    )

            step_i += 1

            if terminated or truncated:
                print("Episode ended. Press R to reset.")
                time.sleep(0.2)

            clock.tick(30)

    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
