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
from duckiematrix_env import DuckiematrixDB21JEnv
from reward_wrappers import LaneFollowingRewardWrapper
from lane_utils import pose_to_lane_frame, LANE_CHECK_MODE, LANE_POS_METHOD

SHOW_POSE = True
LANE_REWARD_KWARGS = {
    "reward_mode": "posangle",
    "include_velocity_reward": True,
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

def _fmt_scalar(value):
    if value is None:
        return "None"
    try:
        return f"{float(value): .3f}"
    except Exception:
        return str(value)


def _fmt_pose_xyz(pose):
    if pose is None:
        return "None"
    x = float(pose["position"]["x"])
    y = float(pose["position"]["y"])
    z = float(pose["position"]["z"])
    return f"({x:.3f},{y:.3f},{z:.3f})"


def _fmt_point(p):
    p = np.asarray(p, dtype=np.float32)
    return f"({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f})"


def _fmt_seq(values):
    if values is None:
        return "None"
    try:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        return "[" + ",".join(f"{float(v):.3f}" for v in arr) + "]"
    except Exception:
        return str(values)


def _current_tile_control_points(base_env):
    pose = getattr(base_env, "last_pose", None)
    if pose is None:
        return None, None, []

    pos_lane, _ = lu.pose_to_lane_frame(pose, lp_cal=base_env.lp_cal)
    tile, _, _ = lu.tile_info(base_env.lp_cal, pos_lane)
    tile_obj = base_env.lp_cal.map_interpreter.get_tile(int(tile[0]), int(tile[1]))
    raw_curves = [] if tile_obj is None else tile_obj.get("curves", [])
    lane_curves = lu._apply_curve_offset(base_env.lp_cal, raw_curves) or []

    curves_out = []
    for idx, cps in enumerate(lane_curves):
        cps_arr = np.asarray(cps, dtype=np.float32)
        labels = ",".join(f"p{i}={_fmt_point(p)}" for i, p in enumerate(cps_arr))
        curves_out.append(f"c{idx}[{labels}]")
    return tile, tile_obj, curves_out


def _print_respawn_snapshot(base_env, label: str = "Respawn"):
    pose = getattr(base_env, "last_pose", None)
    if pose is None:
        print(f"{label}: pose=None")
        return

    try:
        pos_lane, yaw_lane = lu.pose_to_lane_frame(pose, lp_cal=base_env.lp_cal)
    except Exception as e:
        print(f"{label}: pose={_fmt_pose_xyz(pose)} lane_frame_error={type(e).__name__}")
        return

    tile, tile_obj, _ = _current_tile_control_points(base_env)
    tile_kind = None if tile_obj is None else tile_obj.get("kind")

    try:
        lp = lu.get_lane_pos(base_env.lp_cal, pos_lane, float(yaw_lane))
        dist = float(lp.dist)
        dot = float(lp.dot_dir)
        angle_deg = float(lp.angle_deg)
        print(
            f"{label}: pose={_fmt_pose_xyz(pose)} yaw={_fmt_scalar(yaw_lane)} "
            f"tile={tile} kind={tile_kind} dist={dist: .3f} abs_dist={abs(dist): .3f} "
            f"dot={dot: .3f} angle_deg={angle_deg: .3f}"
        )
    except Exception as e:
        print(
            f"{label}: pose={_fmt_pose_xyz(pose)} yaw={_fmt_scalar(yaw_lane)} "
            f"tile={tile} kind={tile_kind} lane_pos_error={type(e).__name__}"
        )

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
    _print_respawn_snapshot(base_env, label="Initial respawn")
    if SHOW_POSE and base_env.last_pose is not None:
        pos0, yaw0 = pose_to_lane_frame(base_env.last_pose, lp_cal=base_env.lp_cal)
        print(f"Initial pose={_fmt_pose_xyz(base_env.last_pose)} yaw={_fmt_scalar(yaw0)}")

    pygame.init()
    pygame.display.set_mode((420, 120))
    pygame.display.set_caption(f"Lane test: mode={LANE_CHECK_MODE}")
    clock = pygame.time.Clock()

    print("\nControls: WASD/Arrows, SPACE brake, R reset, ESC quit\n")
    print(f"Lane rule: mode={LANE_CHECK_MODE} -> lane_pos={LANE_POS_METHOD}\n")

    step_i = 0
    last_print = 0.0
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
                _print_respawn_snapshot(base_env, label="Manual respawn")
                step_i = 0
                total_reward = 0.0
                time.sleep(0.15)
                continue

            action = keys_to_action(keys)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            dist = None
            dot = None
            dbg = {}
            if isinstance(info, dict):
                dist = info.get("lp_dist")
                dot = info.get("lp_dot_dir")
                dbg = dict(info.get("debug_lane_reward", {}) or {})

            now = time.time()
            if now - last_print > 0.10:
                last_print = now
                pose = getattr(base_env, "last_pose", None)
                wheel_vels = getattr(base_env, "wheelVels", None)
                yaw_lane = None
                if pose is not None:
                    try:
                        _, yaw_lane = lu.pose_to_lane_frame(pose, lp_cal=base_env.lp_cal)
                    except Exception:
                        yaw_lane = None
                tile, tile_obj, curves_out = _current_tile_control_points(base_env)
                tile_kind = None if tile_obj is None else tile_obj.get("kind")
                print(f"[{step_i:05d}] pose={_fmt_pose_xyz(pose)} yaw={_fmt_scalar(yaw_lane)} tile={tile} kind={tile_kind}")
                print(
                    "  action_reward: "
                    f"action_wheels={_fmt_seq(action)} wheels={_fmt_seq(wheel_vels)} "
                    f"base_reward={_fmt_scalar(dbg.get('base_reward'))} "
                    f"orientation={_fmt_scalar(dbg.get('orientation_reward'))} "
                    f"velocity={_fmt_scalar(dbg.get('velocity_reward'))} "
                    f"final_reward={float(reward): .3f} total={total_reward: .3f}"
                )
                print(
                    "  lane_terms: "
                    f"mode={LANE_CHECK_MODE} method={LANE_POS_METHOD} "
                    f"dist={_fmt_scalar(dist)} dot={_fmt_scalar(dot)} "
                    f"angle_deg={_fmt_scalar(dbg.get('lp_angle_deg'))} "
                    f"target_angle_deg={_fmt_scalar(dbg.get('target_angle_deg'))} "
                    f"speed={_fmt_scalar(dbg.get('speed'))}"
                )
                print(
                    "  status: "
                    f"terminated={terminated} truncated={truncated} "
                    f"terminated_from_env={dbg.get('terminated_from_env')} "
                    f"pose_valid={dbg.get('pose_valid')} "
                    f"invalid_points={dbg.get('invalid_points')} "
                    f"first_invalid={dbg.get('first_invalid_point')} "
                    f"reasons={dbg.get('reasons')}"
                )
                print(f"  control_points={curves_out}")

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
