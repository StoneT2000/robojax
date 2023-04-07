"""
Wrappers for Brax and Gym env. Code adapted from https://github.com/google/brax/blob/main/brax/envs/wrappers.py
"""

from typing import ClassVar, Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
from brax.envs import env as brax_env
from chex import PRNGKey
from gymnax.environments import spaces

State = brax_env.State


class BraxGymWrapper(gym.Env):
    """A wrapper that converts Brax Env to one that follows Gym API with state. Adapted from Brax's original implementation"""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: brax_env.Env, backend: Optional[str] = None, max_episode_steps: int = -1, auto_reset=True):
        self._env = env
        self.auto_reset = auto_reset
        self.max_episode_steps = max_episode_steps
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
        }
        self.backend = backend

        obs_high = jnp.inf * jnp.ones(self._env.observation_size, dtype="float32")
        self._observation_space = spaces.Box(-obs_high, obs_high, shape=obs_high.shape, dtype="float32")

        action_high = jnp.ones(self._env.action_size, dtype="float32")
        self._action_space = spaces.Box(-action_high, action_high, shape=action_high.shape, dtype="float32")

        def reset(key):
            state = self._env.reset(key)
            return state.obs, state

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            return (
                state,
                state.obs,
                state.reward,
                state.done,
                {**state.metrics, **state.info},
            )

        self._step = jax.jit(step, backend=self.backend)

    def action_space(self, params=None) -> spaces.Discrete:
        """Action space of the environment."""
        return self._action_space

    def observation_space(self, params=None) -> spaces.Box:
        """Observation space of the environment."""
        return self._observation_space

    def reset(self, rng_reset_key: PRNGKey):
        obs, state = self._reset(rng_reset_key)
        state: brax_env.State
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["steps"] = 0
        return obs, state, {}

    def step(self, rng_key, state: brax_env.State, action):
        # rng_key is not used as env is deterministic
        state = state.replace(done=0)
        state, obs, reward, done, info = self._step(state, action)
        steps = state.info["steps"] + 1
        truncated = jnp.where(steps >= self.max_episode_steps, True, False)
        terminated = done != 0.0

        done = terminated | truncated

        info["final_observation"] = obs
        info["_final_observation"] = done

        pipeline_state = jax.tree_map(
            lambda x, y: jnp.where(done, x, y), state.info["first_pipeline_state"], state.pipeline_state
        )
        obs = jnp.where(done, state.info["first_obs"], obs)
        steps = jnp.where(done, 0, steps)
        state.info["steps"] = steps
        state = state.replace(pipeline_state=pipeline_state, obs=obs)

        return obs, state, reward, terminated, truncated, info

    def render(self, state: State, mode="human"):
        # pylint:disable=g-import-not-at-top
        from brax.io import image

        if mode == "rgb_array":
            sys, qp = self._env.sys, state.qp
            return image.render_array(sys, qp, 256, 256)
        else:
            return super().render(mode=mode)  # just raise an exception
