import logging
import time
from types import MethodType

import numpy as np
from duckietown.sdk.utils.lane_position import LanePositionCalculator, MapInterpreter, gen_rot_matrix

import lane_utils as lu

DEFAULT_CAPTURE_TIMEOUT_S = 2.0
DEFAULT_CAPTURE_RETRIES = 3
DEFAULT_CAPTURE_RETRY_SLEEP_S = 0.2


logger = logging.getLogger(__name__)


def _stream_context(self) -> str:
    entity_name = getattr(self, "entity_name", "unknown")
    host = getattr(self, "host", "unknown")
    port = getattr(self, "port", "unknown")
    return f"entity={entity_name} host={host} port={port}"


def _capture_with_retries(
    self,
    *,
    stream_name: str,
    capture_fn,
    timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S,
    retries: int = DEFAULT_CAPTURE_RETRIES,
):
    retries = max(1, int(retries))
    timeout_s = max(0.1, float(timeout_s))
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            data = capture_fn(block=True, timeout=timeout_s)
        except Exception as e:
            last_error = e
            data = None
        if data is not None:
            if attempt > 1:
                logger.warning(
                    "%s stream recovered after %d/%d attempts (%s)",
                    stream_name,
                    attempt,
                    retries,
                    _stream_context(self),
                )
            return data
        if attempt < retries:
            time.sleep(DEFAULT_CAPTURE_RETRY_SLEEP_S * attempt)

    details = _stream_context(self)
    if last_error is not None:
        raise RuntimeError(
            f"{stream_name} stream stalled after {retries} attempts x {timeout_s:.1f}s "
            f"({details}): {last_error}"
        ) from last_error
    raise RuntimeError(
        f"{stream_name} stream stalled: no fresh {stream_name.lower()} available after "
        f"{retries} attempts x {timeout_s:.1f}s ({details})."
    )


def _patched_get_pose_blocking(self, timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S):
    return _capture_with_retries(
        self,
        stream_name="Pose",
        capture_fn=self.robot.pose.capture,
        timeout_s=timeout_s,
    )


def _patched_get_rgb_frame_blocking(self, timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S):
    bgr = _capture_with_retries(
        self,
        stream_name="Camera",
        capture_fn=self.robot.camera.capture,
        timeout_s=timeout_s,
    )
    rgb = np.asarray(bgr, dtype=np.uint8)[:, :, [2, 1, 0]]
    return rgb


def _patched_lane_reward_fn(self, pose):

    assert self.last_pose is not None, "Last pose is None in reward function! Make sure the environment was reset initially."
    last_pose = self.last_pose

    pos_lane, yaw = lu.pose_to_lane_frame(pose, lp_cal=self.lp_cal)
    pose_valid = lu.is_valid_pose(
        self.lp_cal,
        pos_lane,
        float(yaw),
        safety_factor=lu.FOOTPRINT_SAFETY,
    )
    if not pose_valid:
        return float(self.out_of_road_penalty), True

    delta_t = float(pose["header"]["timestamp"]) - float(last_pose["header"]["timestamp"])
    x = float(pose["position"]["x"])
    y = float(pose["position"]["y"])
    z = float(pose["position"]["z"])
    last_x = float(last_pose["position"]["x"])
    last_y = float(last_pose["position"]["y"])
    last_z = float(last_pose["position"]["z"])
    dx = x - last_x
    dy = y - last_y
    dz = z - last_z
    speed = np.sqrt(dx * dx + dy * dy + dz * dz) / delta_t if delta_t > 0 else 0.0

    try:
        lp = lu.get_lane_pos(self.lp_cal, pos_lane, float(yaw))
        reward = +1.0 * speed * float(lp.dot_dir) + -10.0 * abs(float(lp.dist))
    except Exception:
        return 0.0, False

    return float(reward), False


def _patched_step(self, actions):
    actions = np.asarray(actions, dtype=np.float32)
    wl = float(actions[0])
    wr = float(actions[1])
    self.wheelVels = np.array([wl, wr], dtype=np.float32)
    self.robot.motors.set_pwm(left=wl, right=wr)

    obs = self._get_rgb_frame_blocking()
    pose = self._get_pose_blocking()

    reward, terminated = self.reward_fn(pose)
    self.last_pose = pose
    info = self._get_info()
    self._draw_obs(obs)
    truncated = False

    return obs, reward, terminated, truncated, info


def _apply_reward_fn_patch(env) -> None:
    env.reward_fn = MethodType(_patched_lane_reward_fn, env)


def _apply_step_patch(env) -> None:
    env.step = MethodType(_patched_step, env)


def _apply_capture_patch(env) -> None:
    env._get_pose_blocking = MethodType(_patched_get_pose_blocking, env)
    env._get_rgb_frame_blocking = MethodType(_patched_get_rgb_frame_blocking, env)


def use_patched_map_interpreter(env) -> None:
    """
    Replace the environment's curve interpreter and blocking step/reward hooks.
    """
    env.map_int = PatchedMapInterpreter(map=env.map)
    env.lp_cal = LanePositionCalculator(map_interpreter=env.map_int)
    _apply_capture_patch(env)
    _apply_step_patch(env)
    _apply_reward_fn_patch(env)


class PatchedMapInterpreter(MapInterpreter):
    def _get_curve(self, i: int, j: int) -> np.ndarray:
        tile = self.get_tile(i, j)
        assert tile is not None

        kind = tile["kind"]
        angle = float(tile["angle"])

        templates = {
            # Adjust straight and curve templates while leaving 3-way tiles unchanged.
            "straight": [
                [[-0.20, 0, 0.50], [-0.20, 0, 0.25], [-0.20, 0, -0.25],[-0.20, 0, -0.50]],
                [[ 0.20, 0,-0.50],[ 0.20, 0,-0.25],[ 0.20, 0,  0.25], [ 0.20, 0,  0.50]],
            ],
            "curve": [
                [[-0.50, 0, -0.20], [0.00, 0, -0.20], [0.20, 0,  0.00],[0.20, 0, 0.50]],
                [[-0.20, 0,0.50], [-0.20, 0,0.30], [ -0.30, 0, 0.20],[ -0.50, 0, 0.20] ],
            ],
            "3way": [
                [[-0.20, 0, -0.50], [-0.20, 0, -0.25], [-0.20, 0, 0.25], [-0.20, 0, 0.50]],
                [[ 0.20, 0,  0.50], [ 0.20, 0,  0.25], [ 0.20, 0,-0.25], [ 0.20, 0,-0.50]],
                [[ 0.50, 0,  0.20], [ 0.25, 0,  0.20], [-0.25, 0, 0.20], [-0.50, 0, 0.20]],
            ],
        }

        if kind not in templates:
            return np.zeros((0, 4, 3), dtype=np.float32)

        pts = np.array(templates[kind], dtype=np.float32) * float(self.road_tile_size)

        mat = gen_rot_matrix(np.array([0, 1, 0], dtype=np.float32), angle)
        pts = np.matmul(pts, mat)

        pose = tile["pose"]
        pts += np.array(
            [float(pose["x"]+0.5) * self.road_tile_size, 0.0, float(pose["y"]+0.5) * self.road_tile_size],
            dtype=np.float32,
        )
        return pts
