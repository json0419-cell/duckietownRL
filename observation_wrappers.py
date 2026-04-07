import gymnasium as gym
import numpy as np
import cv2


DEFAULT_CAMERA_FOV_X_DEG = 95.0
DEFAULT_MOTION_BLUR_TIME_S = 0.05
DEFAULT_MOTION_BLUR_REFERENCE_KERNEL = 3.0


def _find_unwrapped(env):
    current = env
    seen = set()
    while True:
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)

        if hasattr(current, "env") and getattr(current, "env") is not None:
            current = current.env
            continue
        if hasattr(current, "unwrapped"):
            try:
                current = current.unwrapped
                continue
            except Exception:
                pass
        break
    return current


def _quaternion_to_yaw(rotation: dict) -> float | None:
    try:
        w = float(rotation["w"])
        x = float(rotation["x"])
        y = float(rotation["y"])
        z = float(rotation["z"])
    except Exception:
        return None

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _angle_diff(x: float, y: float) -> float:
    raw = float(y) - float(x)
    return float(np.arctan2(np.sin(raw), np.cos(raw)))


class ResizeCropWrapper(gym.ObservationWrapper):
    def __init__(self, env, out_h=80, out_w=160, crop_top_ratio=0.33):
        super().__init__(env)
        self.out_h = out_h
        self.out_w = out_w
        self.crop_top_ratio = crop_top_ratio

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(out_h, out_w, 3), dtype=np.uint8
        )

    def observation(self, obs):
        H = obs.shape[0]
        top = int(H * self.crop_top_ratio)
        img = obs[top:, :, :]
        img = cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return img


class RandomFrameRepeatingWrapper(gym.ObservationWrapper):
    def __init__(self, env, repeat_prob=0.0):
        super().__init__(env)
        self.repeat_prob = float(np.clip(repeat_prob, 0.0, 0.999))
        self.previous_frame = None

    def observation(self, observation):
        if self.previous_frame is None:
            self.previous_frame = np.array(observation, copy=True)
            return observation

        if np.random.random() < self.repeat_prob:
            return np.array(self.previous_frame, copy=True)

        self.previous_frame = np.array(observation, copy=True)
        return observation

    def reset(self, **kwargs):
        self.previous_frame = None
        return super().reset(**kwargs)


class MotionBlurWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        kernel_size=0,
        camera_fov_x_deg=DEFAULT_CAMERA_FOV_X_DEG,
        blur_time_s=DEFAULT_MOTION_BLUR_TIME_S,
    ):
        super().__init__(env)
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 0:
            raise ValueError("kernel_size must be non-negative")
        self.camera_fov_x_rad = np.deg2rad(float(camera_fov_x_deg))
        self.blur_time_s = float(blur_time_s)
        if self.camera_fov_x_rad <= 0.0:
            raise ValueError("camera_fov_x_deg must be positive")
        if self.blur_time_s <= 0.0:
            raise ValueError("blur_time_s must be positive")

        self.kernel_strength = float(self.kernel_size) / DEFAULT_MOTION_BLUR_REFERENCE_KERNEL
        self.fallback_kernel_size = self._normalize_kernel_size(self.kernel_size)
        self.previous_yaw = None
        self.previous_timestamp = None

    @staticmethod
    def _normalize_kernel_size(kernel_size: int) -> int:
        size = int(kernel_size)
        if size > 0 and size % 2 == 0:
            size += 1
        return size

    @staticmethod
    def _horizontal_kernel(kernel_size: int) -> np.ndarray | None:
        normalized = MotionBlurWrapper._normalize_kernel_size(kernel_size)
        if normalized <= 1:
            return None

        kernel = np.zeros((normalized, normalized), dtype=np.float32)
        kernel[normalized // 2, :] = 1.0 / float(normalized)
        return kernel

    def _current_pose_state(self) -> tuple[float | None, float | None]:
        base_env = _find_unwrapped(self.env)
        pose = getattr(base_env, "last_pose", None)
        if not isinstance(pose, dict):
            return None, None

        try:
            timestamp = float(pose["header"]["timestamp"])
        except Exception:
            timestamp = None
        yaw = _quaternion_to_yaw(pose.get("rotation", {}))
        return yaw, timestamp

    def _dynamic_kernel_size(self, observation) -> int | None:
        current_yaw, current_timestamp = self._current_pose_state()
        if current_yaw is None or current_timestamp is None:
            return None

        if self.previous_yaw is None or self.previous_timestamp is None:
            self.previous_yaw = current_yaw
            self.previous_timestamp = current_timestamp
            return 1

        dt = float(current_timestamp) - float(self.previous_timestamp)
        if dt <= 1e-6:
            self.previous_yaw = current_yaw
            self.previous_timestamp = current_timestamp
            return 1

        angular_velocity = _angle_diff(self.previous_yaw, current_yaw) / dt
        self.previous_yaw = current_yaw
        self.previous_timestamp = current_timestamp

        delta_angle = abs(float(angular_velocity)) * self.blur_time_s
        width_px = int(np.asarray(observation).shape[1])
        ksize = int(np.round(delta_angle / self.camera_fov_x_rad * width_px * self.kernel_strength)) + 1
        if width_px > 1:
            max_ksize = width_px if width_px % 2 == 1 else width_px - 1
            ksize = min(ksize, max_ksize)
        return self._normalize_kernel_size(ksize)

    def observation(self, observation):
        dynamic_kernel_size = self._dynamic_kernel_size(observation)
        if dynamic_kernel_size is not None:
            kernel = self._horizontal_kernel(dynamic_kernel_size)
        else:
            kernel = self._horizontal_kernel(self.fallback_kernel_size)

        if kernel is None:
            return observation

        blurred = cv2.filter2D(observation, -1, kernel)
        return blurred

    def reset(self, **kwargs):
        self.previous_yaw = None
        self.previous_timestamp = None
        obs, info = self.env.reset(**kwargs)
        current_yaw, current_timestamp = self._current_pose_state()
        self.previous_yaw = current_yaw
        self.previous_timestamp = current_timestamp
        return obs, info
