import os
import sys
import time

import numpy as np
import pygame

# Allow running from the test/ directory without installing the project.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import lane_utils as lu
from map_interpreter_patch import use_patched_map_interpreter
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from reward_wrappers import LaneFollowingRewardWrapper
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
LANE_REWARD_KWARGS = {
    "reward_mode": "posangle",
    "include_velocity_reward": True,
    "include_collision_avoidance": True,
}



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

def _collect_curves(lp_cal, pos):
    curves = []
    if lu.NEIGHBOR_TILE_RADIUS > 0:
        for t in lu._iter_neighbor_tiles(lp_cal, pos):
            t_curves = t.get("curves")
            if t_curves is None:
                continue
            curves.extend(lu._apply_curve_offset(lp_cal, t_curves))
    else:
        (i, j), _, _ = lu.tile_info(lp_cal, pos)
        tile = lp_cal.map_interpreter.get_tile(i, j)
        if tile is not None and tile.get("drivable", False):
            curves = lu._apply_curve_offset(lp_cal, tile.get("curves"))
    return curves if curves else None

def _closest_curve_point(lp_cal, pos, angle):
    curves = _collect_curves(lp_cal, pos)
    if not curves:
        return None, None
    cps = lu._select_curve_by_heading(curves, angle) if angle is not None else None
    if cps is None:
        best = None
        best_t = None
        best_abs = None
        for cand in curves:
            t = lu._bezier_closest(cand, pos)
            point = lu._bezier_point(cand, t)
            d = float(np.linalg.norm(point - pos))
            if best is None or d < best_abs:
                best = cand
                best_t = t
                best_abs = d
        cps = best
        t = best_t
    else:
        t = lu._bezier_closest(cps, pos)
    if cps is None or t is None:
        return None, None
    point = lu._bezier_point(cps, t)
    tangent = lu._bezier_tangent(cps, t)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        return point, None
    return point, tangent / norm

def main():
    base_env = DuckiematrixDB21JEnv(
        entity_name="map_0/vehicle_0",
        out_of_road_penalty=-10.0,
        headless=False,
        camera_height=480,
        camera_width=640,
    )
    use_patched_map_interpreter(base_env)
    env = LaneFollowingRewardWrapper(base_env, **LANE_REWARD_KWARGS)
    env.reset()
    if SHOW_POSE and base_env.last_pose is not None:
        pos0, yaw0 = pose_to_lane_frame(base_env.last_pose, lp_cal=base_env.lp_cal)
        print(f"Initial {_fmt_pose(base_env.last_pose, yaw0)}")

    pygame.init()
    pygame.display.set_mode((420, 120))
    pygame.display.set_caption(f"Lane test: mode={LANE_CHECK_MODE}")
    clock = pygame.time.Clock()

    print("\nControls: WASD/Arrows, SPACE brake, R reset, ESC quit\n")
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

            action = keys_to_action(keys)
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

                if dir_dot is not None:
                    dir_info = f"  dir_dot={dir_dot: .3f}  dir_ok={dir_ok}"
                else:
                    dir_info = ""


                lp = None
                lp_err = None
                try:
                    lp = get_lane_pos(base_env.lp_cal, pos, yaw_use)
                except Exception as e:
                    lp_err = e

                if lp is not None:
                    if LANE_CHECK_MODE == "center":
                        half = lane_threshold(base_env.lp_cal, pos=pos)
                        lp_info = f"dist={float(lp.dist): .3f}  half={half: .3f}  dot_dir={float(lp.dot_dir): .3f}"
                    else:
                        lp_info = f"dist={float(lp.dist): .3f}  dot_dir={float(lp.dot_dir): .3f}"
                else:
                    lp_info = f"lp_error={type(lp_err).__name__}"

                dbg_info = ""
                if dbg is not None:
                    reasons = ",".join(dbg.get("reasons", []))
                    dbg_info = (
                        f"  DBG(mode={dbg.get('reward_mode')!r}, speed={dbg.get('speed')!r}, "
                        f"dist={dbg.get('lp_dist')!r}, dot={dbg.get('lp_dot_dir')!r}, "
                        f"angle={dbg.get('lp_angle_deg')!r}, r_orient={dbg.get('orientation_reward')!r}, "
                        f"r_vel={dbg.get('velocity_reward')!r}, r_coll={dbg.get('collision_avoidance_reward')!r}, "
                        f"done_env={dbg.get('terminated_from_env')!r}, reasons={reasons})"
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
