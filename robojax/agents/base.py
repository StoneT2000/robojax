import time
from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey

from robojax.data.loop import EnvAction, EnvObs, EnvState, GymLoop, JaxLoop, BaseEnvLoop
from robojax.logger.logger import Logger
from robojax.models.model import Params
from robojax.utils.spaces import get_action_dim, get_obs_shape


class BasePolicy:
    def __init__(self, jax_env: bool, env=None, eval_env=None, logger_cfg: Any = dict()) -> None:
        """
        Base class for a policy

        Equips it with loopers
        """
        assert env is not None
        self.jax_env = jax_env
        if jax_env:
            self.env_step: Callable[
                [PRNGKey, EnvState, EnvAction],
                Tuple[EnvObs, EnvState, float, bool, Any],
            ] = env.step
            self.env_reset: Callable[[PRNGKey], Tuple[EnvObs, EnvState]] = env.reset
            import gymnax.environments.environment

            self.env: gymnax.environments.environment.Environment = env

            self.loop = JaxLoop(env_reset=self.env.reset, env_step=self.env.step)
            self.observation_space = self.env.observation_space()
            self.action_space = self.env.action_space()
        else:
            import gym

            self.env: gym.Env = env

            self.loop = GymLoop(self.env)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        self.obs_shape = get_obs_shape(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)

        # auto generate an experiment name based on the environment name and current time
        if "exp_name" not in logger_cfg:
            exp_name = f"sac/{round(time.time_ns() / 1000)}"
            if hasattr(env, "name"):
                exp_name = f"{env.name}/{exp_name}"
            logger_cfg["exp_name"] = exp_name

        self.logger = Logger(**logger_cfg)

    @property
    def total_env_steps(self):
        """
        Total number of environment steps run so far
        """
        raise NotImplementedError()
    def evaluate(
        self,
        rng_key: PRNGKey,
        num_envs: int,
        steps_per_env: int,
        eval_loop: BaseEnvLoop,
        params: Params,
        apply_fn: Callable[[PRNGKey, EnvObs], EnvAction],
    ):
        """
        Evaluation function that uses an evaluation loop and executes the apply_fn policy with the given params

        Runs `num_envs * steps_per_env` total steps, split across `num_envs` envs

        Will use the provided logger and store the evaluation returns, episode lengths, and log it all.
        """
        rng_key, *eval_rng_keys = jax.random.split(rng_key, num_envs + 1)
        eval_buffer, _ = eval_loop.rollout(
            rng_keys=jnp.stack(eval_rng_keys),
            params=params,
            apply_fn=apply_fn,
            steps_per_env=steps_per_env,
        )
        eval_ep_lens = np.asarray(eval_buffer["ep_len"])
        eval_ep_rets = np.asarray(eval_buffer["ep_ret"])
        eval_episode_ends = np.asarray(eval_buffer["done"])
        eval_ep_rets = eval_ep_rets[eval_episode_ends].flatten()
        eval_ep_lens = eval_ep_lens[eval_episode_ends].flatten()
        self.logger.store(
            tag="test",
            ep_ret=eval_ep_rets,
            ep_len=eval_ep_lens,
            append=False,
        )
        self.logger.log(self.total_env_steps)
        self.logger.reset()
