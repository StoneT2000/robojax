"""
Environment Loops
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Callable, List, Tuple, TypeVar

import gym
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey

EnvObs = TypeVar("EnvObs")
EnvState = TypeVar("EnvState")
EnvAction = TypeVar("EnvAction")


class BaseEnvLoop(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def rollout(
        self, rng_keys: List[PRNGKey], apply_fn: Callable, steps_per_env: int
    ) -> None:
        raise not NotImplementedError("Rollout not defined")


class GymLoop(BaseEnvLoop):
    """
    RL loop with an environment
    """

    def __init__(self, env: gym.Env, rollout_callback: Callable = None) -> None:
        self.env = env
        self.rollout_callback = rollout_callback
        super().__init__()

    def rollout(self, rng_keys: List[PRNGKey], apply_fn: Callable, steps_per_env: int):
        """
        perform a rollout on a non jittable environment
        """
        num_envs = len(rng_keys)
        rng_key = rng_keys[-1]
        observations, ep_returns, ep_lengths = (
            self.env.reset(),
            np.zeros(num_envs),
            np.zeros(num_envs, dtype=int),
        )
        data = defaultdict(list)
        for t in range(steps_per_env):
            rng_key, rng_fn_key = jax.random.split(rng_key)
            actions, aux = apply_fn(rng_fn_key, observations)
            next_observations, rewards, dones, infos = self.env.step(actions)
            ep_lengths += 1
            ep_returns += rewards
            epoch_ended = t == steps_per_env - 1
            if self.rollout_callback is not None:
                rb = self.rollout_callback(
                    action=actions,
                    env_obs=observations,
                    reward=rewards,
                    ep_ret=ep_returns,
                    ep_len=ep_lengths,
                    next_env_obs=next_observations,
                    done=dones,
                    info=infos,
                    aux=aux,
                )
            else:
                rb = [observations, actions, rewards, next_observations, dones]
            for k, v in rb.items():
                data[k].append(v)
            observations = next_observations
            for idx, terminal in enumerate(dones):
                if terminal or epoch_ended:
                    ep_returns[idx] = 0
                    ep_lengths[idx] = 0

        # stack data
        for k in data:
            data[k] = jnp.stack(data[k])
        return data


class JaxLoop(BaseEnvLoop):
    """
    Env loop for jax based environments
    """
    def __init__(
        self,
        env_reset: Callable[[PRNGKey], Tuple[EnvObs, EnvState]],
        env_step: Callable[
            [PRNGKey, EnvState, EnvAction],
            Tuple[EnvObs, EnvState, float, bool, Any],
        ],
        rollout_callback: Callable = None,
    ) -> None:
        self.env_reset = env_reset
        self.env_step = env_step
        self.rollout_callback = rollout_callback
        super().__init__()

    @partial(jax.jit, static_argnames=["self", "steps", "apply_fn"])
    def _rollout_single_env(
        self,
        rng_key: PRNGKey,
        apply_fn: Callable,
        steps: int,
    ):
        """
        Rollsout on a single env
        """
        rng_key, reset_rng_key = jax.random.split(rng_key)
        env_obs, env_state = self.env_reset(reset_rng_key)

        def step_fn(data: Tuple[EnvObs, EnvState, float, int], _):
            rng_key, env_obs, env_state, ep_ret, ep_len = data
            rng_key, rng_reset, rng_step, rng_fn = jax.random.split(rng_key, 4)
            action, aux = apply_fn(rng_fn, env_obs)
            next_env_obs, next_env_state, reward, done, info = self.env_step(
                rng_step, env_state, action
            )

            def episode_end_update(ep_ret, ep_len, env_state, env_obs):
                env_obs, env_state = self.env_reset(rng_reset)
                return ep_ret * 0, ep_len * 0, env_state, env_obs

            def episode_mid_update(ep_ret, ep_len, env_state, env_obs):
                return ep_ret + reward, ep_len + 1, env_state, env_obs

            new_ep_return, new_ep_len, next_env_state, next_env_obs = jax.lax.cond(
                done,
                episode_end_update,
                episode_mid_update,
                ep_ret,
                ep_len,
                next_env_state,
                next_env_obs,
            )
            # new_ep_return = ep_ret + reward
            # new_ep_len = ep_len + 1

            if self.rollout_callback is not None:
                rb = self.rollout_callback(
                    action=action,
                    env_obs=env_obs,
                    reward=reward,
                    ep_ret=ep_ret,
                    ep_len=ep_len,
                    next_env_obs=next_env_obs,
                    done=done,
                    info=info,
                    aux=aux,
                )
            else:
                rb = [env_obs, action, reward, next_env_obs, done]
            return (
                rng_key,
                next_env_obs,
                next_env_state,
                new_ep_return,
                new_ep_len,
            ), rb

        step_init = (rng_key, env_obs, env_state, jnp.zeros((1,)), jnp.zeros((1,)))
        _, rollout_data = jax.lax.scan(step_fn, step_init, (), steps)
        return rollout_data

    @partial(jax.jit, static_argnames=["self", "steps_per_env", "apply_fn"])
    def rollout(
        self,
        rng_keys: List[PRNGKey],
        apply_fn: Callable,
        steps_per_env: int,
    ):
        """
        Rolls out on len(rng_keys) parallelized environments with a given policy and returns a
        buffer produced by rollout_callback

        This rollout style vmaps rollouts on each parallel env, which are all continuous.
        Once an episode is done, the next immediately starts


        Note: This is faster than only jitting an episode rollout and using a valid mask to remove
        time steps in rollouts that occur after the environment is done.
        The speed increase is noticeable for low number of parallel envs

        The downside here is that the first run will always take extra time but this is
        generally quite minimal overhead over the long term.

        """
        batch_rollout = jax.vmap(
            self._rollout_single_env, in_axes=(0, None, None), out_axes=(1)
        )
        return batch_rollout(jnp.stack(rng_keys), apply_fn, steps_per_env)
