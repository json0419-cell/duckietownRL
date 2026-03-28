"""
Query out-of-bounds and lane-position information from a running environment.

This file can be imported as a helper module or run directly to print pose,
tile, control-point, and lane-offset information continuously.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

# Allow running this file directly from the test/ directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import lane_utils as lu
from map_interpreter_patch import use_patched_map_interpreter


def _pose_xyz(pose: Dict[str, Any]) -> Dict[str, float]:
    return {
        "x": float(pose["position"]["x"]),
        "y": float(pose["position"]["y"]),
        "z": float(pose["position"]["z"]),
    }


def _tile_size(lp_cal) -> Optional[float]:
    size = getattr(lp_cal, "road_tile_size", None)
    if size is None:
        mi = getattr(lp_cal, "map_interpreter", None)
        size = getattr(mi, "road_tile_size", None)
    if size is None:
        return None
    try:
        return float(size)
    except Exception:
        return None


def _pose_yaw(pose: Dict[str, Any]) -> float:
    # Keep yaw extraction consistent with the environment's reward logic.
    from gym_duckiematrix.utils import quaternion_to_euler

    quat_rot = [
        float(pose["rotation"]["w"]),
        float(pose["rotation"]["x"]),
        float(pose["rotation"]["y"]),
        float(pose["rotation"]["z"]),
    ]
    _, _, yaw = quaternion_to_euler(quat_rot)
    return float(yaw)


def _lane_point_to_sdk_point(p_lane: np.ndarray) -> np.ndarray:
    """
    Convert a lane-frame point back into the control-template coordinate order.

    The input uses the current lane frame, and the output follows the local
    control-point template ordering.
    """
    p = np.asarray(p_lane, dtype=np.float32)
    if lu.LANE_UP_AXIS == "z":
        sign = float(lu.LANE_Z_SIGN) if float(lu.LANE_Z_SIGN) != 0.0 else 1.0
        return np.array([p[0], p[2] / sign, p[1]], dtype=np.float32)
    return np.array([p[0], p[1], p[2]], dtype=np.float32)


def _inverse_patch_local_from_lane_point(
    p_lane: np.ndarray,
    tile_pose: Dict[str, Any],
    tile_size: Optional[float],
):
    """
    Convert a world-space point back into the tile-local control-point frame.
    """
    if tile_size is None or tile_size <= 0.0 or tile_pose is None:
        return None, None

    from duckietown.sdk.utils.lane_position import gen_rot_matrix

    p_sdk = _lane_point_to_sdk_point(np.asarray(p_lane, dtype=np.float32))
    yaw = float(tile_pose.get("yaw", 0.0))
    offset = np.array(
        [
            (float(tile_pose.get("x", 0.0)) + 0.5) * float(tile_size),
            0.0,
            (float(tile_pose.get("y", 0.0)) + 0.5) * float(tile_size),
        ],
        dtype=np.float32,
    )
    rel = p_sdk - offset
    rot_inv = gen_rot_matrix(np.array([0.0, 1.0, 0.0], dtype=np.float32), -yaw)
    local_metric = np.matmul(rel, rot_inv)
    local_unit = local_metric / float(tile_size)
    return (
        [float(local_unit[0]), float(local_unit[1]), float(local_unit[2])],
        [float(local_metric[0]), float(local_metric[1]), float(local_metric[2])],
    )


def get_oob_info(env, safety_factor: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Read tile, control-point, tangent, and lane-offset data for the current pose.

    Args:
      env: Environment object that exposes `last_pose` and `lp_cal`.
      safety_factor: Kept for interface compatibility with pose validity checks.

    Returns:
      None if `env.last_pose` is unavailable yet, else a dict with:
        - timestamp
        - pose_xyz
        - tile_size
        - bot_tile
        - bot_tile_kind
        - bot_tile_drivable
        - bot_local_pos
        - bot_tile_all_control_world
        - bot_tile_all_control_local
        - control_world_pos
        - control_tile
        - control_local_pos
        - dist
        - dot
    """
    pose = getattr(env, "last_pose", None)
    if pose is None:
        return None

    _ = safety_factor
    pose_xyz = _pose_xyz(pose)
    pos_lane, yaw_lane = lu.pose_to_lane_frame(pose, lp_cal=env.lp_cal)
    pos_lane = np.asarray(pos_lane, dtype=np.float32)
    yaw = float(yaw_lane)
    tile_size = _tile_size(env.lp_cal)
    dist = None
    dot = None
    angle_deg = None
    angle_rad = None
    bot_tile = None
    bot_tile_kind = None
    bot_tile_drivable = None
    bot_local_pos = None
    bot_local_pos_m = None
    bot_tile_all_control_world = []
    bot_tile_all_control_local = []
    bot_tile_all_control_local_m = []
    control_world_pos = None
    control_tile = None
    control_local_pos = None
    control_local_pos_m = None
    control_tangent = None
    lane_pos_error = None
    control_points_error = None
    try:
        bot_grid, bot_kind, bot_drivable = lu.tile_info(env.lp_cal, pos_lane)
        bot_tile = [int(bot_grid[0]), int(bot_grid[1])]
        bot_tile_kind = bot_kind
        bot_tile_drivable = bool(bot_drivable)

        tile = env.lp_cal.map_interpreter.get_tile(int(bot_grid[0]), int(bot_grid[1]))
        tile_pose = None if tile is None else tile.get("pose")
        bot_local_pos, bot_local_pos_m = _inverse_patch_local_from_lane_point(pos_lane, tile_pose, tile_size)

        raw_curves = [] if tile is None else tile.get("curves", [])
        lane_curves = lu._apply_curve_offset(env.lp_cal, raw_curves) or []
        for cps in lane_curves:
            cps_arr = np.asarray(cps, dtype=np.float32)
            world_pts = []
            local_pts = []
            local_pts_m = []
            for p in cps_arr:
                p_arr = np.asarray(p, dtype=np.float32)
                world_pts.append([float(p_arr[0]), float(p_arr[1]), float(p_arr[2])])
                lp_u, lp_m = _inverse_patch_local_from_lane_point(p_arr, tile_pose, tile_size)
                local_pts.append(lp_u)
                local_pts_m.append(lp_m)
            bot_tile_all_control_world.append(world_pts)
            bot_tile_all_control_local.append(local_pts)
            bot_tile_all_control_local_m.append(local_pts_m)
    except Exception as e:
        control_points_error = f"{type(e).__name__}: {e}"

    try:
        lp = lu.get_lane_pos(env.lp_cal, pos_lane, yaw)
        dist = float(lp.dist)
        dot = float(lp.dot_dir)
        angle_deg = float(lp.angle_deg)
        angle_rad = float(lp.angle_rad)

        point, tangent = lu._closest_curve_point(env.lp_cal, pos_lane, yaw)
        if tangent is not None:
            tan = np.asarray(tangent, dtype=np.float32)
            control_tangent = [float(tan[0]), float(tan[1]), float(tan[2])]
        point_arr = np.asarray(point, dtype=np.float32)
        control_world_pos = [
            float(point_arr[0]),
            float(point_arr[1]),
            float(point_arr[2]),
        ]
        tile_grid, _, _ = lu.tile_info(env.lp_cal, point_arr)
        control_tile = [int(tile_grid[0]), int(tile_grid[1])]
        tile2 = env.lp_cal.map_interpreter.get_tile(int(tile_grid[0]), int(tile_grid[1]))
        tile2_pose = None if tile2 is None else tile2.get("pose")
        control_local_pos, control_local_pos_m = _inverse_patch_local_from_lane_point(
            point_arr, tile2_pose, tile_size
        )
    except Exception as e:
        lane_pos_error = f"{type(e).__name__}: {e}"

    return {
        "timestamp": float(pose["header"]["timestamp"]),
        "pose_xyz": pose_xyz,
        "yaw": yaw,
        "tile_size": tile_size,
        "bot_tile": bot_tile,
        "bot_tile_kind": bot_tile_kind,
        "bot_tile_drivable": bot_tile_drivable,
        "bot_local_pos": bot_local_pos,
        "bot_local_pos_m": bot_local_pos_m,
        "bot_tile_all_control_world": bot_tile_all_control_world,
        "bot_tile_all_control_local": bot_tile_all_control_local,
        "bot_tile_all_control_local_m": bot_tile_all_control_local_m,
        "control_world_pos": control_world_pos,
        "control_tile": control_tile,
        "control_local_pos": control_local_pos,
        "control_local_pos_m": control_local_pos_m,
        "control_tangent": control_tangent,
        "dist": dist,
        "dot": dot,
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
        "control_points_error": control_points_error,
        "lane_pos_error": lane_pos_error,
    }


def format_oob_info(info: Dict[str, Any]) -> str:
    """
    Format the state payload as a compact single-line log string.
    """
    xyz = info["pose_xyz"]
    yaw = info.get("yaw")
    if yaw is None:
        pose_str = f"pose=({xyz['x']:.3f},{xyz['y']:.3f},{xyz['z']:.3f}) yaw=None"
    else:
        pose_str = f"pose=({xyz['x']:.3f},{xyz['y']:.3f},{xyz['z']:.3f}) yaw={yaw:.3f}"
    tile_size = info.get("tile_size")
    tile_str = "tile_size=None" if tile_size is None else f"tile_size={tile_size:.3f}"
    bt = info.get("bot_tile")
    bt_str = "bot_tile=None" if bt is None else f"bot_tile=({bt[0]},{bt[1]})"
    btk_str = f"bot_tile_kind={info.get('bot_tile_kind')}"
    btd_str = f"bot_tile_drivable={info.get('bot_tile_drivable')}"
    btl = info.get("bot_local_pos")
    if btl is None:
        btl_str = "bot_local_pos=None"
    else:
        btl_str = f"bot_local_pos=({btl[0]:.3f},{btl[1]:.3f},{btl[2]:.3f})"
    btlm = info.get("bot_local_pos_m")
    if btlm is None:
        btlm_str = "bot_local_pos_m=None"
    else:
        btlm_str = f"bot_local_pos_m=({btlm[0]:.3f},{btlm[1]:.3f},{btlm[2]:.3f})"
    all_ctrl_world = info.get("bot_tile_all_control_world") or []
    all_ctrl_local = info.get("bot_tile_all_control_local") or []
    ctrl_chunks = []
    for ci, cps in enumerate(all_ctrl_world):
        if len(cps) >= 4:
            p0, p1, p2, p3 = cps[0], cps[1], cps[2], cps[3]
            ctrl_chunks.append(
                f"c{ci}["
                f"p0=({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f}),"
                f"p1=({p1[0]:.3f},{p1[1]:.3f},{p1[2]:.3f}),"
                f"p2=({p2[0]:.3f},{p2[1]:.3f},{p2[2]:.3f}),"
                f"p3=({p3[0]:.3f},{p3[1]:.3f},{p3[2]:.3f})]"
            )
        else:
            pts = ",".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in cps)
            ctrl_chunks.append(f"c{ci}[{pts}]")
    ctrl_world_str = "tile_control_world=[]" if not ctrl_chunks else f"tile_control_world={ctrl_chunks}"

    ctrl_local_chunks = []
    for ci, cps in enumerate(all_ctrl_local):
        safe_pts = [p for p in cps if p is not None]
        if len(safe_pts) >= 4:
            p0, p1, p2, p3 = safe_pts[0], safe_pts[1], safe_pts[2], safe_pts[3]
            ctrl_local_chunks.append(
                f"c{ci}["
                f"p0=({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f}),"
                f"p1=({p1[0]:.3f},{p1[1]:.3f},{p1[2]:.3f}),"
                f"p2=({p2[0]:.3f},{p2[1]:.3f},{p2[2]:.3f}),"
                f"p3=({p3[0]:.3f},{p3[1]:.3f},{p3[2]:.3f})]"
            )
        elif len(safe_pts) > 0:
            pts = ",".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in safe_pts)
            ctrl_local_chunks.append(f"c{ci}[{pts}]")
    ctrl_local_str = "tile_control_local=[]" if not ctrl_local_chunks else f"tile_control_local={ctrl_local_chunks}"
    angle_deg = info.get("angle_deg")
    angle_rad = info.get("angle_rad")
    if angle_deg is None or angle_rad is None:
        angle_str = "angle_deg=None angle_rad=None"
    else:
        angle_str = f"angle_deg={angle_deg:.3f} angle_rad={angle_rad:.3f}"

    if info["dist"] is not None and info["dot"] is not None:
        return (
            f"{pose_str} "
            f"{tile_str} "
            f"{bt_str} "
            f"{btk_str} "
            f"{btd_str} "
            f"{btl_str} "
            f"{btlm_str} "
            f"{ctrl_world_str} "
            f"{ctrl_local_str} "
            f"{angle_str} "
            f"dist={info['dist']:.3f} "
            f"dot={info['dot']:.3f}"
        )
    return (
        f"{pose_str} "
        f"{tile_str} "
        f"{bt_str} "
        f"{btk_str} "
        f"{btd_str} "
        f"{btl_str} "
        f"{btlm_str} "
        f"{ctrl_world_str} "
        f"{ctrl_local_str} "
        f"{angle_str} "
        "dist=None dot=None "
        f"error={info.get('lane_pos_error')}"
    )


def _run_cli():
    parser = argparse.ArgumentParser(description="Print pose, tile, and lane/control-point info.")
    parser.add_argument("--entity-name", type=str, default="map_0/vehicle_0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--poll-hz", type=float, default=10.0, help="print frequency")
    parser.add_argument("--safety-factor", type=float, default=1.0)
    parser.add_argument("--step-env", action="store_true", default=True, help="call env.step([wl,wr]) each loop")
    parser.add_argument("--no-step-env", dest="step_env", action="store_false", help="do not step env")
    parser.add_argument("--wl", type=float, default=0.0, help="left wheel action for --step-env")
    parser.add_argument("--wr", type=float, default=0.0, help="right wheel action for --step-env")
    parser.add_argument("--print-unchanged", action="store_true", help="print even if timestamp unchanged")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after N loop iterations (-1: infinite)")
    parser.add_argument("--auto-reset", action="store_true", help="reset env automatically when terminated/truncated")
    args = parser.parse_args()

    from duckiematrix_env import DuckiematrixDB21JEnv

    env = DuckiematrixDB21JEnv(
        entity_name=args.entity_name,
        out_of_road_penalty=-10.0,
        headless=args.headless,
        camera_height=480,
        camera_width=640,
    )
    use_patched_map_interpreter(env)
    env.reset()

    sleep_dt = 0.1 if args.poll_hz <= 0 else 1.0 / args.poll_hz
    action = np.array([args.wl, args.wr], dtype=np.float32)
    last_t = None

    print("Running oob_info monitor. Ctrl+C to stop.")
    loop_i = 0
    try:
        while True:
            if args.step_env:
                _, _, terminated, truncated, _ = env.step(action)
                if (terminated or truncated) and args.auto_reset:
                    env.reset()

            info = get_oob_info(env, safety_factor=float(args.safety_factor))
            if info is None:
                print("waiting_pose=True")
            else:
                t = info["timestamp"]
                if args.print_unchanged or (last_t is None or t != last_t):
                    print(format_oob_info(info))
                last_t = t
            time.sleep(sleep_dt)
            loop_i += 1
            if args.max_steps >= 0 and loop_i >= args.max_steps:
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    _run_cli()
