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
# Lane math supports two vertical-axis conventions:
# - "y": x/z are the ground plane and y is up
# - "z": x/y are the ground plane and z is up
LANE_UP_AXIS = "z"
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
# Default to footprint-based out-of-bounds checks:
# a pose is invalid once the center, side, or front checkpoints leave drivable tiles.
LANE_CHECK_MODE = "footprint"  # "tile", "footprint", or "center"
FOOTPRINT_SAFETY = 1.0
# Label kept for debug scripts; the implementation always uses one lane-position workflow.
LANE_POS_METHOD = "gym"
LANE_WIDTH = 0.23
LANE_WIDTH_TILE_SCALE = 0.4
LANE_WIDTH_SCALE = 1.0
LANE_WIDTH_MARGIN = 0.0
# Legacy knobs kept for older debug scripts; the current implementation ignores them.
CURVE_OFFSET_TILES = 0.0
CURVE_EXTRA_MARGIN = 0.0
NEIGHBOR_TILE_RADIUS = 0
HEADING_MIN_DOT = -1.0
# If diagnostic left/right labels appear flipped, enable this to swap labels only.
# This does not change the underlying validity logic.
DIAGNOSTIC_SWAP_LEFT_RIGHT = True


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


def _lane_is_y_up():
    if LANE_UP_AXIS not in ("y", "z"):
        raise ValueError(f"Unknown LANE_UP_AXIS: {LANE_UP_AXIS}")
    return LANE_UP_AXIS == "y"


def _up_axis_vec():
    if _lane_is_y_up():
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _to_lane_vec(vec, pose_is_z_up):
    if _lane_is_y_up():
        if pose_is_z_up:
            return np.array([vec[0], vec[2], LANE_Z_SIGN * vec[1]], dtype=np.float32)
        return np.array([vec[0], vec[1], vec[2]], dtype=np.float32)

    # Handle coordinate layouts where z is the vertical axis.
    if pose_is_z_up:
        return np.array([vec[0], vec[1], LANE_Z_SIGN * vec[2]], dtype=np.float32)
    return np.array([vec[0], vec[2], LANE_Z_SIGN * vec[1]], dtype=np.float32)


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
    if _lane_is_y_up():
        yaw = YAW_SIGN * float(math.atan2(-f_lane[2], f_lane[0]))
    else:
        yaw = YAW_SIGN * float(math.atan2(f_lane[1], f_lane[0]))

    if _lane_is_y_up():
        if pose_is_z_up:
            pos_lane = np.array([x, z, LANE_Z_SIGN * y], dtype=np.float32)
        else:
            pos_lane = np.array([x, y, z], dtype=np.float32)
    else:
        if pose_is_z_up:
            pos_lane = np.array([x, y, LANE_Z_SIGN * z], dtype=np.float32)
        else:
            pos_lane = np.array([x, z, LANE_Z_SIGN * y], dtype=np.float32)
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
    Convert a raw pose into the coordinate convention used by lane math.

    The input and output conventions are controlled by POSE_IS_Z_UP and
    LANE_UP_AXIS.
    """
    pos_lane, yaw = _pose_to_lane(pose, POSE_IS_Z_UP)
    if AUTO_SELECT_AXIS and lp_cal is not None and LANE_CHECK_MODE in ("tile", "center"):
        alt_pos, alt_yaw = _pose_to_lane(pose, not POSE_IS_Z_UP)
        if _drivable_pos(lp_cal, alt_pos) and not _drivable_pos(lp_cal, pos_lane):
            return alt_pos, alt_yaw
    return pos_lane, yaw


def yaw_from_displacement(pos, last_pos, min_dist=1e-4):
    dx = float(pos[0] - last_pos[0])
    if _lane_is_y_up():
        dz = float(pos[2] - last_pos[2])
        if dx * dx + dz * dz < float(min_dist) * float(min_dist):
            return None
        return float(math.atan2(-dz, dx))

    dy = float(pos[1] - last_pos[1])
    if dx * dx + dy * dy < float(min_dist) * float(min_dist):
        return None
    return float(math.atan2(dy, dx))


def _dir_vec(angle):
    if _lane_is_y_up():
        if get_dir_vec is not None:
            return get_dir_vec(angle)
        return np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    return np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)


def _right_vec(angle):
    if _lane_is_y_up():
        if get_right_vec is not None:
            return get_right_vec(angle)
        return np.array([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)

    dir_vec = _dir_vec(angle)
    right = np.cross(dir_vec, _up_axis_vec())
    norm = float(np.linalg.norm(right))
    if norm > 1e-6:
        right = right / norm
    return np.asarray(right, dtype=np.float32)


def _actual_center(pos, angle):
    dir_vec = _dir_vec(angle)
    return pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2.0)) * dir_vec


def _grid_coords(lp_cal, pos):
    if _lane_is_y_up():
        return lp_cal.get_grid_coords(pos)

    tile_size = getattr(lp_cal, "road_tile_size", None)
    if tile_size is None:
        mi = getattr(lp_cal, "map_interpreter", None)
        tile_size = getattr(mi, "road_tile_size", None)
    if tile_size is None or float(tile_size) <= 0.0:
        return lp_cal.get_grid_coords(pos)

    x = float(pos[0])
    y = float(pos[1])
    i = math.floor(x / float(tile_size))
    j = math.floor(y / float(tile_size))
    return int(i), int(j)


def _drivable_pos(lp_cal, pos):
    i, j = _grid_coords(lp_cal, pos)
    tile = lp_cal.map_interpreter.get_tile(i, j)
    return tile is not None and tile.get("drivable", False)


def _raise_not_in_lane(msg):
    if NotInLane is not None:
        raise NotInLane(msg)
    raise Exception(msg)


def tile_info(lp_cal, pos):
    i, j = _grid_coords(lp_cal, pos)
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
    _ = pos
    return _lane_half_width(lp_cal)


def _curve_to_lane_points(cps):
    """
    Convert curve control points into the current lane coordinate frame.

    Control points are provided in the local template ordering. When the
    active convention uses z as the vertical axis, the components are
    rearranged to match the lane-math frame.
    """
    arr = np.asarray(cps, dtype=np.float32)
    if _lane_is_y_up():
        return arr

    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = arr[..., 2]
    out[..., 2] = LANE_Z_SIGN * arr[..., 1]
    return out


def _tile_curves(lp_cal, pos):
    i, j = _grid_coords(lp_cal, pos)
    tile = lp_cal.map_interpreter.get_tile(i, j)
    if tile is None or not tile.get("drivable", False):
        _raise_not_in_lane(f"Point not in lane: {pos}")

    curves = tile.get("curves")
    if curves is None or len(curves) == 0:
        _raise_not_in_lane(f"Point not in lane: {pos}")

    return [_curve_to_lane_points(cps) for cps in curves]


def _closest_curve_point(lp_cal, pos, angle):
    """
    Select a candidate curve by heading and return the nearest point and tangent.
    """
    curves = _tile_curves(lp_cal, pos)

    if angle is None:
        best = None
        best_dist = None
        for cps in curves:
            t = _bezier_closest(cps, pos)
            point = np.asarray(_bezier_point(cps, t), dtype=np.float32)
            tangent = np.asarray(_bezier_tangent(cps, t), dtype=np.float32)
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-6:
                continue
            tangent = tangent / norm
            dist = float(np.linalg.norm(point - pos))
            if best is None or dist < best_dist:
                best_dist = dist
                best = (point, tangent)
        if best is None:
            _raise_not_in_lane(f"Point not in lane: {pos}")
        return best

    dir_vec = _dir_vec(angle)
    valid_curves = [np.asarray(cps, dtype=np.float32) for cps in curves if len(cps) >= 2]
    if len(valid_curves) == 0:
        _raise_not_in_lane(f"Point not in lane: {pos}")

    curve_headings = np.asarray([cps[-1] - cps[0] for cps in valid_curves], dtype=np.float32)
    headings_norm = float(np.linalg.norm(curve_headings))
    if headings_norm < 1e-6:
        _raise_not_in_lane(f"Point not in lane: {pos}")
    curve_headings = curve_headings / headings_norm
    dot_prods = np.dot(curve_headings, dir_vec)
    best_cps = valid_curves[int(np.argmax(dot_prods))]

    if best_cps is None:
        _raise_not_in_lane(f"Point not in lane: {pos}")

    t = _bezier_closest(best_cps, pos)
    point = np.asarray(_bezier_point(best_cps, t), dtype=np.float32)
    tangent = np.asarray(_bezier_tangent(best_cps, t), dtype=np.float32)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        _raise_not_in_lane(f"Invalid lane tangent at point: {pos}")
    tangent = tangent / norm
    return point, tangent


def _apply_curve_offset(lp_cal, curves):
    _ = lp_cal
    if curves is None:
        return None
    return [_curve_to_lane_points(cps) for cps in curves]


def _iter_neighbor_tiles(lp_cal, pos):
    # Compatibility iterator that yields the current tile for older test scripts.
    i, j = _grid_coords(lp_cal, pos)
    tile = lp_cal.map_interpreter.get_tile(i, j)
    if tile is not None and tile.get("drivable", False):
        yield tile


def _select_curve_by_heading(curves, angle):
    if curves is None:
        return None

    curves_list = [np.asarray(cps, dtype=np.float32) for cps in curves]
    if len(curves_list) == 0:
        return None
    if angle is None:
        return curves_list[0]

    valid_curves = [cps for cps in curves_list if cps.shape[0] >= 2]
    if len(valid_curves) == 0:
        return None

    dir_vec = _dir_vec(angle)
    curve_headings = np.asarray([cps[-1, :] - cps[0, :] for cps in valid_curves], dtype=np.float32)
    headings_norm = float(np.linalg.norm(curve_headings))
    if headings_norm < 1e-6:
        return None
    curve_headings = curve_headings / headings_norm
    dot_prods = np.dot(curve_headings, dir_vec)
    return valid_curves[int(np.argmax(dot_prods))]


def is_valid_pose(lp_cal, pos, angle, safety_factor=1.0):
    """
    Check whether the vehicle footprint stays inside the drivable area.

    The function tests the center, left, right, and front checkpoints against
    drivable tiles.
    """
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


def _pose_check_points(pos, angle, safety_factor=1.0):
    """Return the footprint points used by validity checks."""
    if USE_ACTUAL_CENTER:
        center = _actual_center(pos, angle)
    else:
        center = np.asarray(pos, dtype=np.float32)

    f_vec = _dir_vec(angle)
    r_vec = _right_vec(angle)

    l_pos = center - (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
    r_pos = center + (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
    f_pos = center + (safety_factor * 0.5 * ROBOT_LENGTH) * f_vec
    return {
        "center": center,
        "left": l_pos,
        "right": r_pos,
        "front": f_pos,
    }


def _diagnostic_label(name: str) -> str:
    if not DIAGNOSTIC_SWAP_LEFT_RIGHT:
        return name
    if name == "left":
        return "right"
    if name == "right":
        return "left"
    return name


def valid_pose_report(lp_cal, pos, angle, safety_factor=1.0):
    """
    Build a detailed pose validity report.

    Returns:
      valid: bool
      invalid_points: list[str]
      first_invalid_point: str|None
      points: dict(point_name -> {'pos', 'grid', 'kind', 'drivable'})
    """
    points = _pose_check_points(pos, angle, safety_factor=safety_factor)
    point_report = {}
    invalid_points = []

    for raw_name, p in points.items():
        name = _diagnostic_label(raw_name)
        (i, j), kind, drivable = tile_info(lp_cal, p)
        point_report[name] = {
            "pos": np.asarray(p, dtype=np.float32),
            "grid": (int(i), int(j)),
            "kind": kind,
            "drivable": bool(drivable),
        }
        if not drivable:
            invalid_points.append(name)

    valid = len(invalid_points) == 0
    return {
        "valid": valid,
        "invalid_points": invalid_points,
        "first_invalid_point": invalid_points[0] if invalid_points else None,
        "points": point_report,
    }


def out_of_bounds_report(lp_cal, pose, safety_factor=FOOTPRINT_SAFETY):
    """
    Run out-of-bounds checks directly from a raw pose dictionary.
    """
    pos_lane, yaw = pose_to_lane_frame(pose, lp_cal=lp_cal)
    report = valid_pose_report(lp_cal, pos_lane, yaw, safety_factor=safety_factor)
    report["out_of_bounds"] = not report["valid"]
    report["pos_lane"] = pos_lane
    report["yaw"] = float(yaw)
    return report


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
    point, tangent = _closest_curve_point(lp_cal, pos, angle)

    if angle is None:
        dir_vec = tangent
        dot_dir = 1.0
    else:
        dir_vec = _dir_vec(angle)
        dot_dir = float(np.clip(np.dot(dir_vec, tangent), -1.0, 1.0))

    up_vec = _up_axis_vec()
    right_vec = np.asarray(np.cross(tangent, up_vec), dtype=np.float32)
    right_norm = float(np.linalg.norm(right_vec))
    if right_norm < 1e-6:
        _raise_not_in_lane(f"Invalid lane tangent at point: {pos}")
    right_vec = right_vec / right_norm

    signed_dist = float(np.dot(pos - point, right_vec))

    angle_rad = float(math.acos(dot_dir))
    if angle is not None and np.dot(dir_vec, right_vec) < 0.0:
        angle_rad *= -1.0
    angle_deg = float(np.rad2deg(angle_rad))

    return LanePosition(
        dist=signed_dist,
        dot_dir=dot_dir,
        angle_deg=angle_deg,
        angle_rad=angle_rad,
    )
