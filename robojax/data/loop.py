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
from chex import Array, PRNGKey
from robojax.utils import tools

EnvObs = TypeVar("EnvObs")
EnvState = TypeVar("EnvState")
EnvAction = TypeVar("EnvAction")

from flax import struct


@struct.dataclass
class DefaultTimeStep:
    env_obs: EnvObs
    action: EnvAction
    reward: Array
    next_env_obs: EnvObs
    done: bool


@struct.dataclass
class RolloutAux:
    final_env_obs: EnvObs
    final_env_state: EnvState


class BaseEnvLoop(ABC):
    rollout_callback: Callable

    def __init__(self) -> None:
        pass

    @abstractmethod
    def rollout(
        self,
        rng_keys: List[PRNGKey],
        params: Any,
        apply_fn: Callable,
        steps_per_env: int,
        max_episode_length: int = -1,
        init_env_obs_states: Tuple[EnvObs, EnvState] = None,
    ) -> None:
        raise not NotImplementedError("Rollout not defined")


class GymLoop(BaseEnvLoop):
    """
    RL loop for non jittable environments environment
    """

    def __init__(self, env: gym.Env, num_envs: int = 1, rollout_callback: Callable = None) -> None:
        self.env = env
        self.num_envs = num_envs
        self.rollout_callback = rollout_callback
        super().__init__()

    def reset_loop(self, rng_key: PRNGKey):
        obs = self.env.reset()
        return obs, None
    def rollout(
        self,
        rng_keys: List[PRNGKey],
        params: Any,
        apply_fn: Callable,
        steps_per_env: int,
        max_episode_length: int = -1,
        init_env_obs_states: Tuple[EnvObs, EnvState] = None,
    ):
        """
        Rollout across N parallelized non-jitted, non-state parameterized, environments with an actor function apply_fn and
        return the rollout buffer as well as final environment observations and states if available.

        Args :
            rng_key : initial PRNGKey to use for any randomness

            params : any function parameters passed to apply_fn

            apply_fn : a function that takes as input a PRNGKey, params, and an environment observation
                and returns a tuple with the action and any auxilliary data

            steps : number of steps to rollout

            max_episode_length : max number of steps before we truncate the current episode. If -1, we will not truncate any environments

            init_env_obs_states : Initial environment observation and state to step forward from. If None, this calls the given self.env_reset function
                to obtain the initial environment observation and state. If not None, it will not call env.reset first.
        """
        num_envs = len(rng_keys)
        rng_key = rng_keys[-1]
        if init_env_obs_states is None:
            observations = self.env.reset()
        else:
            observations = init_env_obs_states[0]
        
        ep_returns, ep_lengths = (
            np.zeros(num_envs),
            np.zeros(num_envs, dtype=int),
        )
        data = defaultdict(list)
        for t in range(steps_per_env):
            rng_key, rng_fn_key = jax.random.split(rng_key)
            actions, aux = apply_fn(rng_fn_key, params, observations)
            actions = tools.any_to_numpy(actions)
            next_observations, rewards, dones, infos = self.env.step(actions)
            ep_lengths += 1
            ep_returns += rewards
            epoch_ended = t == steps_per_env - 1
            if self.rollout_callback is not None:
                rb = self.rollout_callback(
                    action=actions,
                    env_obs=observations,
                    reward=rewards,
                    ep_ret=ep_returns.copy(),
                    ep_len=ep_lengths.copy(),
                    next_env_obs=next_observations,
                    done=dones,
                    info=infos,
                    aux=aux,
                )
            else:
                rb = dict(
                    env_obs=observations,
                    action=actions,
                    reward=rewards,
                    ep_ret=ep_returns.copy(),
                    ep_len=ep_lengths.copy(),
                    done=dones,
                )
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
        return data, {}


class JaxLoop(BaseEnvLoop):
    """
    Env loop for jax based environments


    Args :
        env_reset : An environment reset function thta takes a PRNGKey and
            returns the initial environment observation and state

        env_step : An environment step function that takes a PRNGKey, state, and action and
            returns the next environment observation, state, reward, done, and info

        reset_env : Whether this env loop class will reset environments when done = True.
            If False, when done = True, the loop only resets recorded episode returns and length.
            The environment itself should have some auto reset functionality.

        rollout_callback : A callback function that takes action, env_obs, reward,
            ep_ret, ep_len, next_env_obs, done, info, aux as input
            note that aux is the auxilliary output of the rolled out policy apply_fn when calling rollout().

            The output of this function is used to create the rollout/replay buffer.

            If this function is set to None, the default generated buffer will contain
            env_obs, action, reward, ep_ret, ep_len, done
    """

    def __init__(
        self,
        env_reset: Callable[[PRNGKey], Tuple[EnvObs, EnvState]],
        env_step: Callable[
            [PRNGKey, EnvState, EnvAction],
            Tuple[EnvObs, EnvState, float, bool, Any],
        ],
        num_envs: int = 1,
        rollout_callback: Callable = None,
        reset_env: bool = True,
    ) -> None:
        self.env_reset = env_reset
        self.env_step = env_step
        self.rollout_callback = rollout_callback
        self.reset_env = reset_env
        self.num_envs = num_envs
        super().__init__()

    @partial(jax.jit, static_argnames=["self"])
    def reset_loop(self, rng_key: PRNGKey):
        rng_keys = jax.random.split(rng_key, self.num_envs + 1)
        rng_key = rng_keys[0]
        obs, states = jax.jit(jax.vmap(self.loop.env_reset))(rng_keys[1:])
        return obs, states

    @partial(jax.jit, static_argnames=["self", "steps", "apply_fn", "max_episode_length"])
    def _rollout_single_env(
        self,
        rng_key: PRNGKey,
        params: Any,
        apply_fn: Callable[[PRNGKey, Any, EnvObs], Tuple[EnvAction, Any]],
        steps: int,
        max_episode_length: int = -1,
        init_env_obs_states: Tuple[EnvObs, EnvState] = None,
    ) -> Tuple[Any, RolloutAux]:
        """
        Rollout on a single env and return the rollout_callback outputs and the last environment observation and state

        Args :
            rng_key : initial PRNGKey to use for any randomness

            params : any function parameters passed to apply_fn

            apply_fn : a function that takes as input a PRNGKey, params, and an environment observation
                and returns a tuple with the action and any auxilliary data

            steps : number of steps to rollout

            max_episode_length : max number of steps before we truncate the current episode. If -1, we will not truncate any environments

            init_env_obs_states : Initial environment observation and state to step forward from. If None, this calls the given self.env_reset function
                to obtain the initial environment observation and state

        """
        rng_key, reset_rng_key = jax.random.split(rng_key)
        if init_env_obs_states is not None:
            env_obs, env_state = init_env_obs_states
        else:
            env_obs, env_state = self.env_reset(reset_rng_key)

        def step_fn(data: Tuple[EnvObs, EnvState, float, int], _):
            rng_key, env_obs, env_state, ep_ret, ep_len = data
            rng_key, rng_reset, rng_step, rng_fn = jax.random.split(rng_key, 4)
            action, aux = apply_fn(rng_fn, params, env_obs)
            next_env_obs, next_env_state, reward, done, info = self.env_step(rng_step, env_state, action)
            done = jax.lax.cond((ep_len == max_episode_length - 1)[0], lambda x: True, lambda x: x, done)

            # auto reset
            def episode_end_update(ep_ret, ep_len, env_state, env_obs):
                if self.reset_env:
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
                # rb = [env_obs, action, reward, next_env_obs, done]
                rb = DefaultTimeStep(
                    env_obs=env_obs,
                    action=action,
                    reward=reward,
                    next_env_obs=next_env_obs,
                    done=done,
                )
            return (
                rng_key,
                next_env_obs,
                next_env_state,
                new_ep_return,
                new_ep_len,
            ), rb

        step_init = (rng_key, env_obs, env_state, jnp.zeros((1,)), jnp.zeros((1,)))
        (_, final_env_obs, final_env_state, _, _), rollout_data = jax.lax.scan(step_fn, step_init, (), steps)

        aux = RolloutAux(final_env_obs=final_env_obs, final_env_state=final_env_state)
        # add batch dimension so it plays nice with vmap in the rollout function
        aux = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), aux)
        return rollout_data, aux

    @partial(
        jax.jit,
        static_argnames=["self", "steps_per_env", "apply_fn", "max_episode_length"],
    )
    def rollout(
        self,
        rng_keys: List[PRNGKey],
        params: Any,
        apply_fn: Callable[[PRNGKey, Any, EnvObs], Tuple[EnvAction, Any]],
        steps_per_env: int,
        max_episode_length: int = -1,
        init_env_obs_states: Tuple[EnvObs, EnvState] = None,
    ) -> Tuple[Any, RolloutAux]:
        """
        Rollout across N parallelized environments with an actor function apply_fn and
        return the rollout buffer as well as final environment observations and states

        Args :
            rng_keys : initial PRNGKeys to use for any randomness. len(rng_keys) is
            the number of parallel environments that will be run

            params : any function parameters passed to apply_fn

            apply_fn : a function that takes as input a PRNGKey, params, and an environment observation
                and returns a tuple with the action and any auxilliary data

            steps_per_env : number of steps to rollout per parallel environment

            max_episode_length : max number of steps before we truncate the current episode. If -1, we will not truncate any environments

            init_env_obs_states : Initial environment observation and state to step forward from.
                If None, this calls the given self.env_reset function
                to obtain the initial environment observation and state


        This rollout style vmaps rollouts on each parallel env. Once an episode is done, the next immediately starts

        Note: This is faster than only jitting an episode rollout and using a valid mask to remove
        time steps in rollouts that occur after the environment is done.
        The speed increase is noticeable for low number of parallel envs

        The downside here is that the first run will always take extra time but this is
        generally quite minimal overhead over the long term.

        """
        batch_rollout = jax.vmap(
            self._rollout_single_env,
            in_axes=(0, None, None, None, None, 0),
            out_axes=(1),
        )
        data, aux = batch_rollout(
            jnp.stack(rng_keys),
            params,
            apply_fn,
            steps_per_env,
            max_episode_length,
            init_env_obs_states,
        )
        aux = jax.tree_util.tree_map(lambda x: x[0], aux)  # remove batch dimension
        return data, aux
