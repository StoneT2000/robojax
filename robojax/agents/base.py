import time
from collections import defaultdict
from typing import Any, Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey

from robojax.data.loop import (
    BaseEnvLoop,
    DefaultTimeStep,
    EnvAction,
    EnvObs,
    EnvState,
    GymLoop,
    JaxLoop,
)
from robojax.logger.logger import Logger
from robojax.models.model import Params
from robojax.utils.spaces import get_action_dim, get_obs_shape


class BasePolicy:
    def __init__(
        self,
        jax_env: bool,
        env=None,
        eval_env=None,
        num_envs: int = 1,
        num_eval_envs: int = 1,
        logger_cfg: Any = None,
    ) -> None:
        """
        Base class for a policy

        Equips it with loopers and loggers
        """
        assert env is not None
        self.jax_env = jax_env
        self.loop: BaseEnvLoop = None
        if jax_env:
            import gymnax.environments.environment

            # TODO see when gymnax upgrades to gymnasium
            self.env: gymnax.environments.environment.Environment = env
            self.env_step: Callable[
                [PRNGKey, EnvState, EnvAction],
                Tuple[EnvObs, EnvState, float, bool, bool, Any],
            ] = self.env.step
            self.env_reset: Callable[[PRNGKey], Tuple[EnvObs, EnvState, Any]] = self.env.reset

            self.loop = JaxLoop(env_reset=self.env.reset, env_step=self.env.step, num_envs=num_envs)
            self.observation_space = self.env.observation_space()
            self.action_space = self.env.action_space()
        else:
            import gymnasium

            self.env: gymnasium.vector.VectorEnv = env

            self.loop = GymLoop(self.env, num_envs=num_envs)
            self.observation_space = self.env.single_observation_space
            self.action_space = self.env.single_action_space
        self.obs_shape = get_obs_shape(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)

        # setup evaluation loop
        self.eval_loop: BaseEnvLoop = None
        if eval_env is not None:
            self.eval_env = eval_env
            if self.jax_env:
                self.eval_loop = JaxLoop(
                    eval_env.reset,
                    eval_env.step,
                )
            else:
                self.eval_loop = GymLoop(eval_env, num_eval_envs)

        # auto generate an experiment name based on the environment name and current time
        if logger_cfg is not None:
            if "exp_name" not in logger_cfg:
                exp_name = f"{round(time.time_ns() / 1000)}"
                if hasattr(env, "name"):
                    exp_name = f"{env.name}/{exp_name}"
                logger_cfg["exp_name"] = exp_name

            if "best_stats_cfg" not in logger_cfg:
                logger_cfg["best_stats_cfg"] = {"test/ep_ret_avg": 1, "train/ep_ret_avg": 1}
            if "save_fn" not in logger_cfg:
                logger_cfg["save_fn"] = self.save
            self.logger = Logger(**logger_cfg)

    def state_dict(self):
        """
        Returns a state dict of this object
        """
        raise NotImplementedError()

    def save(self, save_path: str):
        """
        Save the RL agent, including model states, training states, env states (if possible).
        """
        state_dict = self.state_dict()
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(state_dict))

    def load(self, data):
        raise NotImplementedError

    def load_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            data = flax.serialization.from_bytes(self.state_dict(), f.read())
        self.load(data)
        return self

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
            loop_state=None,  # set to None means this eval_loop will generate its own loop state
            params=params,
            apply_fn=apply_fn,
            steps_per_env=steps_per_env,
        )
        # TODO use eval_buffer info['stats'] to log custom stats
        if not self.jax_env:
            final_infos = eval_buffer["final_info"]
            del eval_buffer["final_info"]
            eval_buffer = DefaultTimeStep(**eval_buffer)
        eval_buffer: DefaultTimeStep = jax.tree_map(lambda x: np.array(x), eval_buffer)
        eval_episode_ends = eval_buffer.truncated | eval_buffer.terminated
        eval_ep_rets = eval_buffer.ep_ret[eval_episode_ends].flatten()
        eval_ep_lens = eval_buffer.ep_len[eval_episode_ends].flatten()
        stats_list = []
        if not self.jax_env:
            # TODO handle final info metrics
            for info in final_infos:
                if "stats" in info:
                    stats_list.append(info["stats"])
        stats = defaultdict(list)
        {stats[key].append(sub[key]) for sub in stats_list for key in sub}
        return dict(eval_ep_rets=eval_ep_rets, eval_ep_lens=eval_ep_lens, stats=stats)
