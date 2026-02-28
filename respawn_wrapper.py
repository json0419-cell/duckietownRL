import gymnasium as gym


class RandomRespawnWrapper(gym.Wrapper):
    """
    在 reset 时随机重生车辆。优先调用底层 env 的 random_respawn(...) 方法；
    如果不存在该方法，则退化为普通 reset（不报错）。

    典型参数：
      lateral_jitter: 车道横向抖动（米）
      yaw_jitter_deg: 朝向抖动（度）
      avoid_junction: 是否避免路口
      fallback_bbox: 备用包围盒（未用到时可为 None）
    """

    def __init__(
        self,
        env: gym.Env,
        lateral_jitter: float = 0.02,
        yaw_jitter_deg: float = 8.0,
        avoid_junction: bool = True,
        fallback_bbox=None,
    ):
        super().__init__(env)
        self.lateral_jitter = lateral_jitter
        self.yaw_jitter_deg = yaw_jitter_deg
        self.avoid_junction = avoid_junction
        self.fallback_bbox = fallback_bbox

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 优先使用底层 env 的 random_respawn（如果可用）
        if hasattr(self.env, "random_respawn"):
            try:
                self.env.random_respawn(
                    lateral_jitter=self.lateral_jitter,
                    yaw_jitter_deg=self.yaw_jitter_deg,
                    avoid_junction=self.avoid_junction,
                    fallback_bbox=self.fallback_bbox,
                )
                # 重新获取观测，因为位置变了
                obs = self.env.render_obs() if hasattr(self.env, "render_obs") else obs
            except Exception:
                # 如果重生失败，保持原 reset 结果
                pass

        return obs, info
