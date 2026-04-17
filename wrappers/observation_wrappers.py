import gymnasium as gym
import numpy as np
import cv2


DEFAULT_CAMERA_FOV_X_DEG = 95.0
DEFAULT_MOTION_BLUR_TIME_S = 0.05
DEFAULT_MOTION_BLUR_REFERENCE_KERNEL = 3.0
DEFAULT_PHOTOMETRIC_AUG_STRENGTH = 0.0
DEFAULT_YELLOW_LANE_AUG_STRENGTH = 0.0
DEFAULT_LANE_MASK_NOISE_STRENGTH = 0.0
DEFAULT_PHOTOMETRIC_BRIGHTNESS_DELTA = 0.18
DEFAULT_PHOTOMETRIC_CONTRAST_DELTA = 0.22
DEFAULT_PHOTOMETRIC_GAMMA_DELTA = 0.22
DEFAULT_PHOTOMETRIC_CHANNEL_GAIN_DELTA = 0.12
DEFAULT_PHOTOMETRIC_SATURATION_DELTA = 0.45
DEFAULT_PHOTOMETRIC_BLUR_SIGMA_MAX = 0.85
DEFAULT_PHOTOMETRIC_JPEG_QUALITY_DELTA = 35
DEFAULT_YELLOW_LANE_SATURATION_DELTA = 0.55
DEFAULT_YELLOW_LANE_BRIGHTNESS_DELTA = 0.35
DEFAULT_YELLOW_LANE_BLUR_SIGMA_MAX = 0.75
DEFAULT_YELLOW_LANE_BLEND_DELTA = 0.45
DEFAULT_LANE_MASK_BINARY_THRESHOLD = 127
DEFAULT_LANE_MASK_RECT_DROPOUT_COUNT = 4
DEFAULT_LANE_MASK_RECT_SIZE = 8
DEFAULT_LANE_MASK_FALSE_POSITIVE_COUNT = 2
DEFAULT_LANE_MASK_PIXEL_DROPOUT = 0.03
DEFAULT_LANE_MASK_MIN_AREA_RATIO = 0.0015
DEFAULT_LANE_MASK_MIN_HEIGHT_RATIO = 0.08
DEFAULT_LANE_MASK_MAX_HORIZONTAL_RATIO = 1.8
DEFAULT_LANE_MASK_MIN_BOTTOM_REACH_RATIO = 0.55
DEFAULT_LANE_MASK_MAX_UPPER_CENTROID_RATIO = 0.38


def _as_uint8_rgb(observation: np.ndarray) -> np.ndarray:
    image = np.asarray(observation)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected image with ndim=2 or 3, got shape={image.shape}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape={image.shape}")
    return image


def _soften_mask(mask: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    softened = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), sigma)
    return np.clip(softened, 0.0, 1.0)


def postprocess_lane_mask(mask: np.ndarray, *, lane_kind: str) -> np.ndarray:
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={binary.shape}")
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)

    h, w = binary.shape
    processed = binary.copy()

    kernel = np.ones((3, 3), dtype=np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    min_area = max(6, int(round(h * w * DEFAULT_LANE_MASK_MIN_AREA_RATIO)))
    min_height = max(5, int(round(h * DEFAULT_LANE_MASK_MIN_HEIGHT_RATIO)))
    bottom_keep_threshold = int(round(h * 0.25))
    min_bottom_reach = int(round(h * DEFAULT_LANE_MASK_MIN_BOTTOM_REACH_RATIO))
    max_upper_centroid = float(h * DEFAULT_LANE_MASK_MAX_UPPER_CENTROID_RATIO)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
    filtered = np.zeros_like(processed)
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        _, cy = centroids[label]
        if area < min_area:
            continue
        if bh < min_height:
            continue
        if bw > max(3, int(round(bh * DEFAULT_LANE_MASK_MAX_HORIZONTAL_RATIO))):
            continue
        if y + bh < bottom_keep_threshold:
            continue

        if lane_kind == "yellow":
            if y + bh < min_bottom_reach and cy < max_upper_centroid and bw > max(4, int(round(bh * 0.65))):
                continue
        else:
            if y + bh < min_bottom_reach and cy < max_upper_centroid:
                continue

        if lane_kind == "yellow" and x > int(round(w * 0.82)):
            continue
        if lane_kind == "white" and x + bw < int(round(w * 0.18)):
            continue

        filtered[labels == label] = 255

    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    return filtered


def extract_yellow_lane_mask(observation: np.ndarray, *, soften: bool = False) -> np.ndarray:
    image = _as_uint8_rgb(observation)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([10, 60, 60], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8),
    )
    lab_mask = cv2.inRange(
        lab,
        np.array([0, 0, 145], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_and(hsv_mask, lab_mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = postprocess_lane_mask(mask, lane_kind="yellow")
    if soften:
        return _soften_mask(mask)
    return mask


def extract_white_lane_mask(observation: np.ndarray, *, soften: bool = False) -> np.ndarray:
    image = _as_uint8_rgb(observation)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 115], dtype=np.uint8),
        np.array([180, 85, 255], dtype=np.uint8),
    )
    lab_mask = cv2.inRange(
        lab,
        np.array([140, 118, 118], dtype=np.uint8),
        np.array([255, 138, 138], dtype=np.uint8),
    )
    mask = cv2.bitwise_and(hsv_mask, lab_mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = postprocess_lane_mask(mask, lane_kind="white")
    if soften:
        return _soften_mask(mask)
    return mask


def combine_lane_masks(yellow_mask: np.ndarray, white_mask: np.ndarray) -> np.ndarray:
    yellow = np.asarray(yellow_mask)
    white = np.asarray(white_mask)
    if yellow.shape != white.shape:
        raise ValueError(f"Yellow/white masks must share shape, got {yellow.shape} and {white.shape}")

    if yellow.dtype != np.uint8:
        yellow = np.clip(yellow, 0.0, 1.0)
        yellow = np.round(yellow * 255.0).astype(np.uint8)
    if white.dtype != np.uint8:
        white = np.clip(white, 0.0, 1.0)
        white = np.round(white * 255.0).astype(np.uint8)
    return np.maximum(yellow, white)


def build_binary_lane_image(
    observation: np.ndarray,
    *,
    merge_yellow_and_white: bool = True,
    out_channels: int = 3,
) -> np.ndarray:
    yellow_mask = extract_yellow_lane_mask(observation, soften=False)
    white_mask = extract_white_lane_mask(observation, soften=False)
    if merge_yellow_and_white:
        combined = combine_lane_masks(yellow_mask, white_mask)
        if out_channels == 1:
            return combined
        return np.repeat(combined[..., None], out_channels, axis=2)

    if out_channels < 2:
        raise ValueError("out_channels must be >= 2 when merge_yellow_and_white is False")
    channels = [yellow_mask, white_mask]
    while len(channels) < out_channels:
        channels.append(np.zeros_like(yellow_mask))
    return np.stack(channels[:out_channels], axis=2)


def apply_lane_mask_noise(
    mask_image: np.ndarray,
    *,
    strength: float = DEFAULT_LANE_MASK_NOISE_STRENGTH,
    rng: np.random.Generator | None = None,
    binary_threshold: int = DEFAULT_LANE_MASK_BINARY_THRESHOLD,
) -> np.ndarray:
    def _apply_2d(mask_2d: np.ndarray) -> np.ndarray:
        noisy = (mask_2d >= int(binary_threshold)).astype(np.uint8) * 255
        h, w = noisy.shape

        kernel_size = 3 if strength >= 0.75 else 2
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        morph_choice = int(rng.integers(0, 4))
        if morph_choice == 0:
            noisy = cv2.erode(noisy, kernel, iterations=1)
        elif morph_choice == 1:
            noisy = cv2.dilate(noisy, kernel, iterations=1)
        elif morph_choice == 2:
            noisy = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
        else:
            noisy = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, kernel)

        dropout_count = int(np.round(DEFAULT_LANE_MASK_RECT_DROPOUT_COUNT * strength))
        max_rect = max(2, int(np.round(DEFAULT_LANE_MASK_RECT_SIZE * max(0.5, strength))))
        for _ in range(dropout_count):
            rect_w = int(rng.integers(2, max_rect + 1))
            rect_h = int(rng.integers(2, max_rect + 1))
            x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
            y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
            noisy[y0 : y0 + rect_h, x0 : x0 + rect_w] = 0

        pixel_dropout_prob = DEFAULT_LANE_MASK_PIXEL_DROPOUT * strength
        if pixel_dropout_prob > 0.0:
            white_pixels = noisy > 0
            random_field = rng.random(noisy.shape)
            noisy[np.logical_and(white_pixels, random_field < pixel_dropout_prob)] = 0

        false_positive_count = int(np.round(DEFAULT_LANE_MASK_FALSE_POSITIVE_COUNT * strength))
        for _ in range(false_positive_count):
            rect_w = int(rng.integers(1, 4))
            rect_h = int(rng.integers(1, 4))
            x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
            y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
            noisy[y0 : y0 + rect_h, x0 : x0 + rect_w] = 255
        return noisy.astype(np.uint8)

    strength = max(0.0, float(strength))
    if strength <= 0.0:
        return np.array(mask_image, copy=True)

    rng = rng if rng is not None else np.random.default_rng()
    image = np.asarray(mask_image)
    if image.ndim == 2:
        return _apply_2d(image)
    if image.ndim != 3:
        raise ValueError(f"Expected mask image with ndim=2 or 3, got shape={image.shape}")
    if image.shape[2] > 1:
        first_channel = image[..., 0]
        if all(np.array_equal(first_channel, image[..., idx]) for idx in range(1, image.shape[2])):
            noisy = _apply_2d(first_channel)
            return np.repeat(noisy[..., None], image.shape[2], axis=2)
    channels = [_apply_2d(image[..., idx]) for idx in range(image.shape[2])]
    return np.stack(channels, axis=2)


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


class BinaryLaneMaskWrapper(gym.ObservationWrapper):
    """Convert RGB observations into a black-background lane mask image."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("BinaryLaneMaskWrapper expects a Box observation space")
        if len(obs_space.shape) != 3:
            raise ValueError(f"Expected HWC observation shape, got {obs_space.shape}")

        h, w, _ = obs_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 3),
            dtype=np.uint8,
        )

    def observation(self, observation):
        lane_image = build_binary_lane_image(observation, merge_yellow_and_white=True, out_channels=3)
        if lane_image.ndim == 2:
            lane_image = lane_image[..., None]
        return lane_image


class LaneMaskNoiseWrapper(gym.ObservationWrapper):
    """Inject imperfections into already-extracted binary lane mask observations."""

    def __init__(self, env, strength: float = DEFAULT_LANE_MASK_NOISE_STRENGTH):
        super().__init__(env)
        self.strength = max(0.0, float(strength))
        self._rng = np.random.default_rng()

    def observation(self, observation):
        return apply_lane_mask_noise(
            observation,
            strength=self.strength,
            rng=self._rng,
        )


class YellowLaneAugWrapper(gym.ObservationWrapper):
    """Sim-only targeted augmentation to weaken the idealized yellow center line."""

    def __init__(self, env, strength=DEFAULT_YELLOW_LANE_AUG_STRENGTH):
        super().__init__(env)
        self.strength = max(0.0, float(strength))
        self.params = None

    def _sample_params(self):
        strength = self.strength
        sat_low = max(0.05, 1.0 - DEFAULT_YELLOW_LANE_SATURATION_DELTA * strength)
        sat_high = max(0.1, 1.0 - 0.12 * strength)
        val_low = max(0.25, 1.0 - DEFAULT_YELLOW_LANE_BRIGHTNESS_DELTA * strength)
        val_high = max(0.4, 1.0 - 0.05 * strength)
        blend_low = 0.35
        blend_high = min(0.95, 0.35 + DEFAULT_YELLOW_LANE_BLEND_DELTA * strength)
        self.params = {
            "saturation_scale": float(np.random.uniform(sat_low, sat_high)),
            "value_scale": float(np.random.uniform(val_low, val_high)),
            "softness_sigma": float(np.random.uniform(0.0, DEFAULT_YELLOW_LANE_BLUR_SIGMA_MAX * strength)),
            "blend": float(np.random.uniform(blend_low, blend_high)),
        }

    @staticmethod
    def _yellow_mask(observation: np.ndarray) -> np.ndarray:
        return extract_yellow_lane_mask(observation, soften=True)

    def observation(self, observation):
        if self.strength <= 0.0:
            return observation
        if self.params is None:
            self._sample_params()

        mask = self._yellow_mask(np.asarray(observation, dtype=np.uint8))
        if float(mask.max()) <= 1e-3:
            return observation

        hsv = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV).astype(np.float32)
        adjusted_hsv = hsv.copy()
        adjusted_hsv[..., 1] *= self.params["saturation_scale"]
        adjusted_hsv[..., 2] *= self.params["value_scale"]
        adjusted_hsv[..., 1:] = np.clip(adjusted_hsv[..., 1:], 0.0, 255.0)

        adjusted_rgb = cv2.cvtColor(adjusted_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        softness_sigma = float(self.params["softness_sigma"])
        if softness_sigma > 1e-3:
            kernel_size = 3 if softness_sigma < 0.6 else 5
            softened = cv2.GaussianBlur(adjusted_rgb, (kernel_size, kernel_size), softness_sigma)
            adjusted_rgb = cv2.addWeighted(adjusted_rgb, 0.65, softened, 0.35, 0.0)

        blend = float(np.clip(self.params["blend"], 0.0, 1.0))
        mask3 = np.repeat(mask[..., None], 3, axis=2) * blend
        output = (
            observation.astype(np.float32) * (1.0 - mask3)
            + adjusted_rgb.astype(np.float32) * mask3
        )
        return np.clip(output, 0.0, 255.0).astype(np.uint8)

    def reset(self, **kwargs):
        self.params = None
        if self.strength > 0.0:
            self._sample_params()
        return super().reset(**kwargs)


class PhotometricAugWrapper(gym.ObservationWrapper):
    def __init__(self, env, strength=DEFAULT_PHOTOMETRIC_AUG_STRENGTH):
        super().__init__(env)
        self.strength = max(0.0, float(strength))
        self.params = None

    def _sample_params(self):
        strength = self.strength
        brightness_low = max(0.05, 1.0 - DEFAULT_PHOTOMETRIC_BRIGHTNESS_DELTA * strength)
        brightness_high = 1.0 + 0.03 * strength
        contrast_low = max(0.05, 1.0 - DEFAULT_PHOTOMETRIC_CONTRAST_DELTA * strength)
        contrast_high = 1.0 - 0.02 * strength
        gamma_low = 1.0 + 0.03 * strength
        gamma_high = 1.0 + DEFAULT_PHOTOMETRIC_GAMMA_DELTA * strength
        channel_low = max(0.05, 1.0 - DEFAULT_PHOTOMETRIC_CHANNEL_GAIN_DELTA * strength)
        channel_high = 1.0 + 0.06 * strength
        saturation_low = max(0.05, 1.0 - DEFAULT_PHOTOMETRIC_SATURATION_DELTA * strength)
        saturation_high = 1.0 - 0.08 * strength

        self.params = {
            "brightness": float(np.random.uniform(brightness_low, brightness_high)),
            "contrast": float(np.random.uniform(contrast_low, contrast_high)),
            "gamma": float(np.random.uniform(gamma_low, gamma_high)),
            "channel_gains": np.random.uniform(channel_low, channel_high, size=3).astype(np.float32),
            "saturation": float(np.random.uniform(saturation_low, saturation_high)),
            "blur_sigma": float(np.random.uniform(0.0, DEFAULT_PHOTOMETRIC_BLUR_SIGMA_MAX * strength)),
            "jpeg_quality": int(
                np.round(np.random.uniform(100 - DEFAULT_PHOTOMETRIC_JPEG_QUALITY_DELTA * strength, 95))
            ),
        }

    def observation(self, observation):
        if self.strength <= 0.0:
            return observation
        if self.params is None:
            self._sample_params()

        img = np.asarray(observation, dtype=np.float32) / 255.0
        img = img * self.params["channel_gains"].reshape(1, 1, 3)
        img = img * self.params["brightness"]
        img = (img - 0.5) * self.params["contrast"] + 0.5
        gray = np.dot(img[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        img = gray[..., None] + self.params["saturation"] * (img - gray[..., None])
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, self.params["gamma"], dtype=np.float32)
        img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)

        blur_sigma = float(self.params["blur_sigma"])
        if blur_sigma > 1e-3:
            kernel_size = 3 if blur_sigma < 0.6 else 5
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), blur_sigma)

        jpeg_quality = int(np.clip(self.params["jpeg_quality"], 30, 100))
        if jpeg_quality < 100:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            ok, encoded = cv2.imencode(".jpg", bgr, encode_params)
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return img

    def reset(self, **kwargs):
        self.params = None
        if self.strength > 0.0:
            self._sample_params()
        return super().reset(**kwargs)


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
