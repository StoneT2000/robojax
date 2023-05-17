"""
Wrappers for Gymnax to make them into the Gymnasium API with added state
"""

from typing import Optional

import gymnasium as gym
import jax
from chex import Array, PRNGKey
from gymnax.environments import spaces
from gymnax.environments.environment import Environment, EnvParams, EnvState


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
            "render.modes": ["human", "rgb_array"],
        }
        self.backend = backend

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
        # TODO verify keeping track of a first state is faster than resetting any time.
        state: EnvState
        # state.info["first_state"] = state
        # state.info["first_obs"] = obs
        # state.info["steps"] = 0
        return obs, state, {}

    def step(self, rng_key: PRNGKey, state: EnvState, action: Array):

        obs, state, reward, done, info = self._step(rng_key, state, action)
        # steps = state.info["steps"] + 1
        truncated = False  # shouldn't always be false but at the moment gymnax does not support proper gymnasium api
        # truncated = jnp.where(steps >= self.max_episode_steps, True, False)
        terminated = done != 0.0

        # info["final_observation"] = obs
        # info["_final_observation"] = done

        # gymnax_state = jax.tree_map(
        #     lambda x, y: jnp.where(done, x, y), state.info["first_state"], state.state
        # )
        # obs = jnp.where(done, state.info["first_obs"], obs)
        # steps = jnp.where(done, 0, steps)
        # state.info["steps"] = steps
        # state = state.replace(state=gymnax_state, obs=obs)

        return obs, state, reward, terminated, truncated, info

    def render(self, state: EnvState, mode="human"):
        # TODO
        raise NotImplementedError()
