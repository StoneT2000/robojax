import gym
import gymnasium
import gymnasium.spaces as spaces
from gymnasium.core import ActType, Env, ObsType


class ManiSkill2Wrapper(gymnasium.Wrapper):
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._action_space = spaces.Box(env.action_space.low, env.action_space.high, env.action_space.shape, env.action_space.dtype)
        self._observation_space = spaces.Box(env.observation_space.low, env.observation_space.high, env.observation_space.shape, env.observation_space.dtype)

    def reset(self,*, seed=None, options=None):
        self.env.seed(seed)
        obs = self.env.reset()
        return obs, {}
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info
    

class ContinuousTaskWrapper(gymnasium.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        terminated = False
        return observation, reward, terminated, truncated, info 