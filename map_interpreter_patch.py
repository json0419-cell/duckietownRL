from types import MethodType

import numpy as np
from duckietown.sdk.utils.lane_position import LanePositionCalculator, MapInterpreter, gen_rot_matrix

import lane_utils as lu

DEFAULT_CAPTURE_TIMEOUT_S = 1.0


def _patched_get_pose_blocking(self, timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S):
    pose = self.robot.pose.capture(block=True, timeout=timeout_s)
    if pose is None:
        raise RuntimeError("Pose stream stalled: no fresh pose available after timeout.")
    return pose


def _patched_get_rgb_frame_blocking(self, timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S):
    bgr = self.robot.camera.capture(block=True, timeout=timeout_s)
    if bgr is None:
        raise RuntimeError("Camera stream stalled: no fresh RGB frame available after timeout.")
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
