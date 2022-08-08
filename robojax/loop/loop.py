from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, List, Tuple, TypeVar
from chex import ArrayTree, PRNGKey
import gym
import jax
import jax.numpy as jnp
import numpy as np
import time

Env_Obs = TypeVar("Env_Obs")
Env_State = TypeVar("Env_State")
Env_Action = TypeVar("Env_Action")


class BaseEnvLoop(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def rollout(self) -> None:
        raise not NotImplementedError("Rollout not defined")


class GymLoop(BaseEnvLoop):
    """
    RL loop with an environment
    """

    def __init__(self) -> None:
        # self.env = env
        pass

    def rollout(self, env: gym.Env, apply_fn: Callable, steps: int, n_envs: int, rollout_callback: Callable = None):
        """
        perform a rollout on a non jittable environment
        """
        rollout_start_time = time.time_ns()
        observations, ep_returns, ep_lengths = env.reset(), np.zeros(n_envs), np.zeros(n_envs, dtype=int)
        for t in range(steps):
            actions, aux = apply_fn(observations)
            next_observations, rewards, dones, infos = env.step(actions)
            ep_lengths += 1
            ep_returns += rewards
            epoch_ended = t == steps - 1
            if rollout_callback is not None:
                rollout_callback(
                    actions=actions,
                    observations=observations,
                    rewards=rewards,
                    ep_returns=ep_returns,
                    ep_lengths=ep_lengths,
                    next_observations=next_observations,
                    dones=dones,
                    infos=infos,
                    aux=aux,
                )
            observations = next_observations
            for idx, terminal in enumerate(dones):
                if terminal or epoch_ended:
                    ep_returns[idx] = 0
                    ep_lengths[idx] = 0
        rollout_end_time = time.time_ns()
        rollout_delta_time = (rollout_end_time - rollout_start_time) * 1e-9


class JaxLoop(BaseEnvLoop):
    def __init__(
        self,
        env_reset: Callable[[PRNGKey], Tuple[Env_Obs, Env_State]],
        env_step: Callable[[PRNGKey, Env_State, Env_Action], Tuple[Env_Obs, Env_State, float, bool, Any]],
        apply_fn: Callable,
        rollout_callback: Callable = None,
    ) -> None:
        self.env_reset = env_reset
        self.env_step = env_step
        self.apply_fn = apply_fn
        self.rollout_callback = rollout_callback
        super().__init__()

    # @partial(jax.jit, static_argnames=["self", "steps"])
    def _rollout_single_env(
        self,
        rng_key: PRNGKey,
        steps: int,
    ):
        """
        Rollsout on a single env
        """
        rng_key, reset_rng_key = jax.random.split(rng_key)
        env_obs, env_state = self.env_reset(reset_rng_key)

        def step_fn(data: Tuple[Env_Obs, Env_State, float, int], i):
            rng_key, env_obs, env_state, ep_ret, ep_len = data
            rng_key, rng_reset, rng_step, rng_fn = jax.random.split(rng_key, 4)
            action, aux = self.apply_fn(rng_fn, env_obs)
            next_env_obs, next_env_state, reward, done, info = self.env_step(rng_step, env_state, action)
            def episode_end_update(ep_ret, ep_len, env_state, env_obs):
                env_obs, env_state = self.env_reset(rng_reset)
                return ep_ret * 0, ep_len * 0, env_state, env_obs

            def episode_mid_update(ep_ret, ep_len, env_state, env_obs):
                return ep_ret + reward, ep_len + 1, env_state, env_obs

            # new_ep_return, new_ep_len, next_env_state, next_env_obs = jax.lax.cond(
            #     done, episode_end_update, episode_mid_update, ep_ret, ep_len, next_env_state, next_env_obs
            # )
            new_ep_return = ep_ret + reward
            new_ep_len = ep_len + 1
            
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
            return (rng_key, next_env_obs, next_env_state, new_ep_return, new_ep_len), rb
        step_init = (rng_key, env_obs, env_state, jnp.zeros((1,)), jnp.zeros((1,)))
        _, rollout_data = jax.lax.scan(step_fn, step_init, (), steps)
        return rollout_data

    @partial(jax.jit, static_argnames=["self", "steps"])
    def rollout(
        self,
        batch_rng_keys: List[PRNGKey],
        steps: int,  # steps per env
    ):
        """
        Rolls out on len(batch_rng_keys) parallelized environments with a given policy and returns a buffer produced by rollout_callback

        This rollout style vmaps rollouts on each parallel env, which are all continuous. Once an episode is done, the next immediately starts
        
        
        Note: This is faster than only jitting an episode rollout and using a valid mask to remove 
        time steps in rollouts that occur after the environment is done. The speed increase is noticeable for low number of parallel envs

        The downside here is that the first run will always take extra time but this is generally quite minimal overhead over the long term.

        """
        batch_rollout = jax.vmap(self._rollout_single_env, in_axes=(0, None))
        return batch_rollout(jnp.stack(batch_rng_keys), steps)