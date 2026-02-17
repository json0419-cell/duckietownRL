import math
from dataclasses import dataclass
import numpy as np

try:
    from duckietown.sdk.utils.lane_position import (
        LanePosition,
        NotInLane,
        get_dir_vec,
        get_right_vec,
        bezier_closest,
        bezier_point,
        bezier_tangent,
    )
except Exception:
    LanePosition = None
    NotInLane = None
    get_dir_vec = None
    get_right_vec = None
    bezier_closest = None
    bezier_point = None
    bezier_tangent = None

if NotInLane is None:
    class NotInLane(Exception):
        pass

if LanePosition is None:
    @dataclass
    class LanePosition:
        dist: float
        dot_dir: float
        angle_deg: float
        angle_rad: float

        def as_json_dict(self):
            return {
                "dist": float(self.dist),
                "dot_dir": float(self.dot_dir),
                "angle_deg": float(self.angle_deg),
                "angle_rad": float(self.angle_rad),
            }


POSE_IS_Z_UP = True
LANE_Z_SIGN = 1.0
YAW_SIGN = 1.0
POSE_SCALE = 1.0
FORWARD_AXIS = None  # "x" for ROS-style, "z" for Unity-style
FORWARD_AXIS_SIGN = 1.0
AUTO_SELECT_AXIS = False
USE_ACTUAL_CENTER = True
CAMERA_FORWARD_DIST = 0.066
ROBOT_LENGTH = 0.18
ROBOT_WIDTH = 0.13 + 0.02
LANE_CHECK_MODE = "center"  # "tile", "footprint", or "center"
FOOTPRINT_SAFETY = 1.0
LANE_POS_METHOD = "distance"  # "heading" or "distance"
LANE_WIDTH = 0.23
LANE_WIDTH_TILE_SCALE = 0.4
LANE_WIDTH_SCALE = 1.0
LANE_WIDTH_MARGIN = 0.0
CURVE_OFFSET_TILES = 0.5
CURVE_EXTRA_MARGIN = 0.02
NEIGHBOR_TILE_RADIUS = 1
HEADING_MIN_DOT = 0.1


def _quat_to_rot_matrix(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _forward_axis(pose_is_z_up):
    if FORWARD_AXIS is not None:
        return FORWARD_AXIS
    return "x" if pose_is_z_up else "z"


def _pose_forward_vec(quat, pose_is_z_up):
    rot = _quat_to_rot_matrix(quat)
    axis = _forward_axis(pose_is_z_up)
    if axis == "x":
        vec = rot[:, 0]
    elif axis == "z":
        vec = rot[:, 2]
    else:
        raise ValueError(f"Unknown FORWARD_AXIS: {axis}")
    return FORWARD_AXIS_SIGN * vec


def _to_lane_vec(vec, pose_is_z_up):
    if pose_is_z_up:
        return np.array([vec[0], vec[2], LANE_Z_SIGN * vec[1]], dtype=np.float32)
    return np.array([vec[0], vec[1], vec[2]], dtype=np.float32)


def _pose_to_lane(pose, pose_is_z_up):
    x = float(pose["position"]["x"])
    y = float(pose["position"]["y"])
    z = float(pose["position"]["z"])
    scale = float(POSE_SCALE)
    if scale != 1.0:
        x *= scale
        y *= scale
        z *= scale

    quat = [
        pose["rotation"]["w"],
        pose["rotation"]["x"],
        pose["rotation"]["y"],
        pose["rotation"]["z"],
    ]
    f_world = _pose_forward_vec(quat, pose_is_z_up)
    f_lane = _to_lane_vec(f_world, pose_is_z_up)
    f_norm = float(np.linalg.norm(f_lane))
    if f_norm > 1e-6:
        f_lane = f_lane / f_norm
    yaw = YAW_SIGN * float(math.atan2(-f_lane[2], f_lane[0]))

    if pose_is_z_up:
        pos_lane = np.array([x, z, LANE_Z_SIGN * y], dtype=np.float32)
    else:
        pos_lane = np.array([x, y, z], dtype=np.float32)
    return pos_lane, yaw


def _bezier_point(cps, t):
    if bezier_point is not None:
        return bezier_point(cps, t)
    cps = np.asarray(cps, dtype=np.float32)
    p = ((1 - t) ** 3) * cps[0, :]
    p += 3 * t * ((1 - t) ** 2) * cps[1, :]
    p += 3 * (t**2) * (1 - t) * cps[2, :]
    p += (t**3) * cps[3, :]
    return p


def _bezier_tangent(cps, t):
    if bezier_tangent is not None:
        return bezier_tangent(cps, t)
    cps = np.asarray(cps, dtype=np.float32)
    p = 3 * ((1 - t) ** 2) * (cps[1, :] - cps[0, :])
    p += 6 * (1 - t) * t * (cps[2, :] - cps[1, :])
    p += 3 * (t**2) * (cps[3, :] - cps[2, :])
    norm = float(np.linalg.norm(p))
    if norm > 1e-6:
        p = p / norm
    return p


def _bezier_closest(cps, p, t_bot=0.0, t_top=1.0, n=8):
    if bezier_closest is not None:
        try:
            return bezier_closest(cps, p, t_bot, t_top, n)
        except TypeError:
            return bezier_closest(cps, p)
    mid = (t_bot + t_top) * 0.5
    if n == 0:
        return mid
    p_bot = _bezier_point(cps, t_bot)
    p_top = _bezier_point(cps, t_top)
    d_bot = float(np.linalg.norm(p_bot - p))
    d_top = float(np.linalg.norm(p_top - p))
    if d_bot < d_top:
        return _bezier_closest(cps, p, t_bot, mid, n - 1)
    return _bezier_closest(cps, p, mid, t_top, n - 1)


def pose_to_lane_frame(pose, lp_cal=None):
    """
    Convert DB21J pose to the lane_position coordinate frame.
    Toggle POSE_IS_Z_UP if your pose uses z-up instead of y-up.
    """
    pos_lane, yaw = _pose_to_lane(pose, POSE_IS_Z_UP)
    if AUTO_SELECT_AXIS and lp_cal is not None and LANE_CHECK_MODE in ("tile", "center"):
        alt_pos, alt_yaw = _pose_to_lane(pose, not POSE_IS_Z_UP)
        if _drivable_pos(lp_cal, alt_pos) and not _drivable_pos(lp_cal, pos_lane):
            return alt_pos, alt_yaw
    return pos_lane, yaw


def yaw_from_displacement(pos, last_pos, min_dist=1e-4):
    dx = float(pos[0] - last_pos[0])
    dz = float(pos[2] - last_pos[2])
    if dx * dx + dz * dz < float(min_dist) * float(min_dist):
        return None
    return float(math.atan2(-dz, dx))


def _dir_vec(angle):
    if get_dir_vec is not None:
        return get_dir_vec(angle)
    return np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)


def _right_vec(angle):
    if get_right_vec is not None:
        return get_right_vec(angle)
    return np.array([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)


def _actual_center(pos, angle):
    dir_vec = _dir_vec(angle)
    return pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2.0)) * dir_vec


def _drivable_pos(lp_cal, pos):
    i, j = lp_cal.get_grid_coords(pos)
    tile = lp_cal.map_interpreter.get_tile(i, j)
    return tile is not None and tile.get("drivable", False)


def _raise_not_in_lane(msg):
    if NotInLane is not None:
        raise NotInLane(msg)
    raise Exception(msg)


def tile_info(lp_cal, pos):
    i, j = lp_cal.get_grid_coords(pos)
    tile = lp_cal.map_interpreter.get_tile(i, j)
    if tile is None:
        return (i, j), None, False
    return (i, j), tile.get("kind"), bool(tile.get("drivable", False))


def _lane_half_width(lp_cal):
    width = None
    if hasattr(lp_cal, "lane_width"):
        try:
            width = float(getattr(lp_cal, "lane_width"))
        except Exception:
            width = None
    if width is None:
        tile_size = getattr(lp_cal, "road_tile_size", None)
        if tile_size is None:
            mi = getattr(lp_cal, "map_interpreter", None)
            tile_size = getattr(mi, "road_tile_size", None)
        if tile_size is not None:
            width = float(tile_size) * float(LANE_WIDTH_TILE_SCALE)
    if width is None or width <= 0.0:
        mi = getattr(lp_cal, "map_interpreter", None)
        if mi is not None and hasattr(mi, "lane_width"):
            try:
                width = float(getattr(mi, "lane_width"))
            except Exception:
                width = None
    if width is None or width <= 0.0:
        width = float(LANE_WIDTH)
    return 0.5 * width * float(LANE_WIDTH_SCALE) + float(LANE_WIDTH_MARGIN)


def lane_half_width(lp_cal):
    return _lane_half_width(lp_cal)


def lane_threshold(lp_cal, pos=None):
    half = _lane_half_width(lp_cal)
    if pos is not None and CURVE_EXTRA_MARGIN != 0.0:
        try:
            i, j = lp_cal.get_grid_coords(pos)
            tile = lp_cal.map_interpreter.get_tile(i, j)
            if tile is not None and tile.get("kind") == "curve":
                half += float(CURVE_EXTRA_MARGIN)
        except Exception:
            pass
    return half


def _curve_offset(lp_cal):
    if CURVE_OFFSET_TILES == 0.0:
        return None
    tile_size = getattr(lp_cal, "road_tile_size", None)
    if tile_size is None:
        mi = getattr(lp_cal, "map_interpreter", None)
        tile_size = getattr(mi, "road_tile_size", None)
    if tile_size is None:
        return None
    offset = float(CURVE_OFFSET_TILES) * float(tile_size)
    return np.array([offset, 0.0, offset], dtype=np.float32)


def _apply_curve_offset(lp_cal, curves):
    offset = _curve_offset(lp_cal)
    if offset is None:
        return curves
    return np.asarray(curves, dtype=np.float32) + offset


def _iter_neighbor_tiles(lp_cal, pos):
    i, j = lp_cal.get_grid_coords(pos)
    radius = int(NEIGHBOR_TILE_RADIUS)
    mi = lp_cal.map_interpreter
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            tile = mi.get_tile(i + di, j + dj)
            if tile is None or not tile.get("drivable", False):
                continue
            yield tile


def _select_curve_by_heading(curves, angle):
    if curves is None:
        return None
    dir_vec = _dir_vec(angle)
    best_cps = None
    best_dot = None
    for cps in curves:
        cps_arr = np.asarray(cps, dtype=np.float32)
        if cps_arr.shape[0] < 2:
            continue
        heading = cps_arr[-1, :] - cps_arr[0, :]
        norm = float(np.linalg.norm(heading))
        if norm < 1e-6:
            continue
        heading = heading / norm
        dot = float(np.dot(heading, dir_vec))
        if best_dot is None or dot > best_dot:
            best_dot = dot
            best_cps = cps_arr
    return best_cps


def is_valid_pose(lp_cal, pos, angle, safety_factor=1.0):
    if USE_ACTUAL_CENTER:
        center = _actual_center(pos, angle)
    else:
        center = pos

    f_vec = _dir_vec(angle)
    r_vec = _right_vec(angle)

    l_pos = center - (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
    r_pos = center + (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
    f_pos = center + (safety_factor * 0.5 * ROBOT_LENGTH) * f_vec

    return (
        _drivable_pos(lp_cal, center)
        and _drivable_pos(lp_cal, l_pos)
        and _drivable_pos(lp_cal, r_pos)
        and _drivable_pos(lp_cal, f_pos)
    )


def is_in_lane(lp_cal, pos, angle):
    mode = LANE_CHECK_MODE
    if mode == "tile":
        return _drivable_pos(lp_cal, pos)
    if mode == "footprint":
        return is_valid_pose(lp_cal, pos, angle, safety_factor=FOOTPRINT_SAFETY)
    if mode == "center":
        try:
            lp = get_lane_pos(lp_cal, pos, angle)
        except Exception:
            return False
        return abs(float(lp.dist)) <= lane_threshold(lp_cal, pos=pos)
    raise ValueError(f"Unknown LANE_CHECK_MODE: {mode}")


def get_lane_pos(lp_cal, pos, angle):
    mode = LANE_POS_METHOD
    if mode == "heading":
        return get_lane_pos_by_heading(lp_cal, pos, angle)
    if mode == "distance":
        return get_lane_pos_by_distance(lp_cal, pos, angle)
    raise ValueError(f"Unknown LANE_POS_METHOD: {mode}")


def get_lane_pos_by_heading(lp_cal, pos, angle):
    """
    Gym-duckietown style: choose curve by heading, then compute lane position.
    """
    i, j = lp_cal.get_grid_coords(pos)
    tile = lp_cal.map_interpreter.get_tile(i, j)
    if tile is None or not tile.get("drivable", False):
        _raise_not_in_lane(f"Point not in lane: {pos}")

    if NEIGHBOR_TILE_RADIUS > 0:
        curves = []
        for t in _iter_neighbor_tiles(lp_cal, pos):
            t_curves = t.get("curves")
            if t_curves is None:
                continue
            curves.extend(_apply_curve_offset(lp_cal, t_curves))
    else:
        curves = _apply_curve_offset(lp_cal, tile.get("curves"))
    if curves is None or len(curves) == 0:
        _raise_not_in_lane(f"Point not in lane: {pos}")

    if angle is None:
        return get_lane_pos_by_distance(lp_cal, pos, angle)

    dir_vec = _dir_vec(angle)
    up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    best = None
    best_abs = None

    for cps in curves:
        t = _bezier_closest(cps, pos)
        point = _bezier_point(cps, t)
        tangent = _bezier_tangent(cps, t)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            continue
        tangent = tangent / norm

        dot_dir = float(np.clip(np.dot(dir_vec, tangent), -1.0, 1.0))
        if dot_dir < float(HEADING_MIN_DOT):
            continue

        right_vec = np.cross(tangent, up_vec)
        signed_dist = float(np.dot(pos - point, right_vec))

        angle_rad = math.acos(dot_dir)
        if np.dot(dir_vec, right_vec) < 0:
            angle_rad *= -1.0
        angle_deg = float(np.rad2deg(angle_rad))

        abs_dist = abs(signed_dist)
        if best is None or abs_dist < best_abs:
            best_abs = abs_dist
            best = LanePosition(
                dist=signed_dist,
                dot_dir=dot_dir,
                angle_deg=angle_deg,
                angle_rad=angle_rad,
            )

    if best is None:
        return get_lane_pos_by_distance(lp_cal, pos, angle)

    return best


def get_lane_pos_by_distance(lp_cal, pos, angle, keep_sign=False):
    """
    Choose the curve with the smallest lateral distance to the agent.
    """

    i, j = lp_cal.get_grid_coords(pos)
    tile = lp_cal.map_interpreter.get_tile(i, j)
    if tile is None or not tile.get("drivable", False):
        _raise_not_in_lane(f"Point not in lane: {pos}")

    curves = []
    if NEIGHBOR_TILE_RADIUS > 0:
        for t in _iter_neighbor_tiles(lp_cal, pos):
            t_curves = t.get("curves")
            if t_curves is None:
                continue
            curves.extend(_apply_curve_offset(lp_cal, t_curves))
    else:
        curves = _apply_curve_offset(lp_cal, tile["curves"])
    if not curves:
        _raise_not_in_lane(f"Point not in lane: {pos}")
    dir_vec = _dir_vec(angle) if angle is not None else None
    up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    best = None
    best_abs = None

    for cps in curves:
        t = _bezier_closest(cps, pos)
        point = _bezier_point(cps, t)
        tangent = _bezier_tangent(cps, t)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            continue
        tangent = tangent / norm

        if dir_vec is None:
            dot_dir = 1.0
        else:
            dot_dir = float(np.clip(np.dot(dir_vec, tangent), -1.0, 1.0))
            if (not keep_sign) and dot_dir < 0.0:
                tangent = -tangent
                dot_dir = -dot_dir

        right_vec = np.cross(tangent, up_vec)
        signed_dist = float(np.dot(pos - point, right_vec))

        angle_rad = math.acos(dot_dir)
        if dir_vec is not None and np.dot(dir_vec, right_vec) < 0:
            angle_rad *= -1.0
        angle_deg = float(np.rad2deg(angle_rad))

        abs_dist = abs(signed_dist)
        if best is None or abs_dist < best_abs:
            best_abs = abs_dist
            best = LanePosition(
                dist=signed_dist,
                dot_dir=dot_dir,
                angle_deg=angle_deg,
                angle_rad=angle_rad,
            )

    if best is None:
        _raise_not_in_lane(f"Point not in lane: {pos}")

    return best
