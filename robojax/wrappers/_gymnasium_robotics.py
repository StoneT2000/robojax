import gymnasium as gym


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        terminated = False
        self.success_once = self.success_once | info["success"]
        info["stats"] = dict(
            success_at_end=int(info["success"]),
            success=self.success_once,
        )
        return observation, reward, terminated, truncated, info
