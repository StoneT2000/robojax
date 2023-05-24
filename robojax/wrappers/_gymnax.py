"""
Wrappers for Gymnax to make them into the Gymnasium API with added state
"""

from typing import List, Optional, Tuple, Union

import chex
import gymnasium as gym
import jax
from chex import Array, PRNGKey
from gym.core import ActType, ObsType, RenderFrame
from gym.vector.utils import batch_space
from gymnax.environments import spaces
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.environments.spaces import gymnax_space_to_gym_space


class GymnaxWrapper(gym.Env):
    """
    A wrapper that converts a Gymnax Env to one that follows Gymnasium API with state

    Note that Gymnax envs auto reset themselves. They call their own reset function every step and return the new obs and state
    so there are no terminal observations by default so this wrapper adds that.
    """

    def __init__(
        self,
        env: Environment,
        env_params: EnvParams,
        backend: Optional[str] = None,
        max_episode_steps: int = -1,
        auto_reset=True,
    ):
        self._env = env
        self.auto_reset = auto_reset
        self.max_episode_steps = max_episode_steps
        self.metadata = {
            "render.modes": ["rgb_array"],
        }
        self.render_mode = "rgb_array"
        self.backend = backend
        self.env_params = env_params
        self._observation_space = env.observation_space(env_params)

        self._action_space = env.action_space(env_params)

        def reset(key):
            obs, state = self._env.reset(key, params=env_params)
            return obs, state

        self._reset = jax.jit(reset, backend=self.backend)

        def step(rng_key: PRNGKey, state: EnvState, action: Array):
            """Performs step transitions in the environment."""
            key, key_reset = jax.random.split(rng_key)
            obs_st, state_st, reward, done, info = self._env.step_env(key, state, action, env_params)
            obs_re, state_re = self._env.reset_env(key_reset, env_params)
            # Auto-reset environment based on termination
            state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
            obs = jax.lax.select(done, obs_re, obs_st)
            info["final_observation"] = obs_st  # this is always the current episodes next observation
            info["_final_observation"] = done  # mask for which observation is current episodes next observation
            return obs, state, reward, done, info

        self._step = jax.jit(step, backend=self.backend)

    def action_space(self, params=None) -> spaces.Discrete:
        """Action space of the environment."""
        return self._action_space

    def observation_space(self, params=None) -> spaces.Box:
        """Observation space of the environment."""
        return self._observation_space

    def reset(self, rng_reset_key: PRNGKey):
        obs, state = self._reset(rng_reset_key)
        return obs, state, {}

    def step(self, rng_key: PRNGKey, state: EnvState, action: Array):

        obs, state, reward, done, info = self._step(rng_key, state, action)
        # steps = state.info["steps"] + 1
        truncated = False  # shouldn't always be false but at the moment gymnax does not support proper gymnasium api
        terminated = done != 0.0
        return obs, state, reward, terminated, truncated, info

    def render(self, state: EnvState, params: EnvParams = None):
        return self._env.render(state)


"""Gymnax Wrapper to Vector Gym Wrapper (No jax). Modified from the origianl Gymnax Repository to conform to Gymnasium API"""


class GymnaxToVectorGymWrapper(gym.vector.VectorEnv):
    def __init__(
        self,
        env: GymnaxWrapper,
        num_envs: int = 1,
        seed: Optional[int] = None,
    ):
        """Wrap Gymnax environment as OOP Gym Vector Environment

        Args:
            env: Gymnax Environment instance
            num_envs: Desired number of environments to run in parallel
            seed: If provided, seed for JAX PRNG (otherwise picks 0)
        """
        self._env = env
        self.num_envs = num_envs
        self.is_vector_env = True
        self.closed = False
        self.viewer = None
        self.render_mode = "rgb_array"

        # Jit-of-vmap is faster than vmap-of-jit.
        self._env.reset = jax.jit(jax.vmap(self._env.reset))
        self._env.step = jax.jit(
            jax.vmap(
                self._env.step,
                in_axes=(
                    0,
                    0,
                    0,
                ),
            )
        )

        self.rng: chex.PRNGKey = jax.random.PRNGKey(0)  # Placeholder
        self._seed(seed)
        _, self.env_state, _ = self._env.reset(
            self.rng,
        )  # Placeholder
        self._batched_rng_split = jax.jit(jax.vmap(jax.random.split, in_axes=0, out_axes=1))  # Split all rng keys

    @property
    def single_action_space(self):
        """Dynamically adjust action space depending on params"""
        return gymnax_space_to_gym_space(self._env.action_space(self._env.env_params))

    @property
    def single_observation_space(self):
        """Dynamically adjust state space depending on params"""
        return gymnax_space_to_gym_space(self._env.observation_space(self._env.env_params))

    @property
    def action_space(self):
        """Dynamically adjust action space depending on params"""
        return batch_space(self.single_action_space, self.num_envs)

    @property
    def observation_space(self):
        """Dynamically adjust state space depending on params"""
        return batch_space(self.single_observation_space, self.num_envs)

    def _seed(self, seed: Optional[int] = None):
        """Set RNG seed (or use 0)"""
        self.rng = jax.random.split(jax.random.PRNGKey(seed or 0), self.num_envs)  # 1 RNG per env

    def reset(self, *, seed: Optional[int] = None, options=None) -> Tuple[ObsType, dict]:
        """Reset environment, update parameters and seed if provided"""
        if seed is not None:
            self._seed(seed)
        self.rng, reset_key = self._batched_rng_split(self.rng)  # Split all keys
        o, self.env_state, info = self._env.reset(reset_key)
        return o, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Step environment, follow new step API"""
        self.rng, step_key = self._batched_rng_split(self.rng)
        obs, self.env_state, reward, terminated, truncated, info = self._env.step(step_key, self.env_state, action)
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """use underlying environment rendering if it exists (for first environment), otherwise return None"""
        return getattr(self._env, "render", lambda x, y: None)(
            jax.tree_map(lambda x: x[0], self.env_state), self._env.env_params
        )
