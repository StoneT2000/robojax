import os
import time
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple

import distrax
import flax
import jax
import numpy as np
from chex import Array, PRNGKey
from flax import struct
from tqdm import tqdm

from robojax.agents.base import BasePolicy
from robojax.agents.sac import loss
from robojax.agents.sac.config import SACConfig, TimeStep
from robojax.agents.sac.networks import ActorCritic, DiagGaussianActor
from robojax.data.buffer import GenericBuffer
from robojax.data.loop import EnvAction
from robojax.models import Model
from robojax.utils import tools


# TODO: Create a algo state / training state with separaable non-jax component (e.g. replay buffer) for easy saving and continuing runs
@struct.dataclass
class TrainStepEnvState:
    env_obs: Array
    env_states: Array

    total_env_steps: int
    training_steps: int

    ep_lens: Array
    ep_rets: Array
    dones: Array


@struct.dataclass
class TrainStepMetrics:
    train_stats: Any
    train: Any
    time: Any


class SAC(BasePolicy):
    def __init__(
        self,
        jax_env: bool,
        ac: ActorCritic,
        num_envs,
        env,
        seed_sampler: Callable[[PRNGKey], EnvAction] = None,
        eval_env=None,
        logger_cfg=dict(),
        cfg: SACConfig = {},
    ):
        super().__init__(jax_env, env, eval_env, num_envs, logger_cfg)
        if isinstance(cfg, dict):
            self.cfg = SACConfig(**cfg)
        else:
            self.cfg = cfg

        self.total_env_steps = 0
        self.training_steps = 0
        self.ac: ActorCritic = ac
        if seed_sampler is None:
            seed_sampler = lambda rng_key: self.env.action_space().sample(rng_key)
            # TODO add a nice error message if this guessed sampler doesn't work
        self.seed_sampler = seed_sampler

        buffer_config = dict(
            action=((self.action_dim,), self.action_space.dtype),
            reward=((), np.float32),
            mask=((), float),
        )
        if isinstance(self.obs_shape, dict):
            buffer_config["env_obs"] = (
                self.obs_shape,
                {k: self.observation_space[k].dtype for k in self.observation_space},
            )
        else:
            buffer_config["env_obs"] = (self.obs_shape, np.float32)
        buffer_config["next_env_obs"] = buffer_config["env_obs"]

        # note that we use GenericBuffer class, which is not backed by jax for storing
        # interactions due to jax being slow for adding small amounts of data and moving data
        # off the GPU
        self.replay_buffer = GenericBuffer(
            buffer_size=self.cfg.replay_buffer_capacity,
            num_envs=self.cfg.num_envs,
            config=buffer_config,
        )

        if self.cfg.target_entropy is None:
            self.cfg.target_entropy = -self.action_dim / 2

        # Jax env specific code to improve speed
        if self.jax_env:
            self._env_step = jax.jit(self._env_step, static_argnames=["seed"])

    @partial(jax.jit, static_argnames=["self", "seed"])
    def _sample_action(self, rng_key, actor: DiagGaussianActor, env_obs, seed=False):
        if seed:
            a = self.seed_sampler(rng_key)
        else:
            dist: distrax.Distribution = actor(env_obs)
            a = dist.sample(seed=rng_key)
        return a

    def _env_step(self, rng_key: PRNGKey, env_obs, env_state, actor: DiagGaussianActor, seed=False):
        rng_key, act_rng_key, env_rng_key = jax.random.split(rng_key, 3)
        a = self._sample_action(act_rng_key, actor, env_obs, seed)
        if self.jax_env:
            (
                next_env_obs,
                next_env_state,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_step(env_rng_key, env_state, a)
        else:
            a = tools.any_to_numpy(a)
            next_env_obs, reward, terminated, truncated, info = self.env.step(a)
            next_env_state = None
        return a, next_env_obs, next_env_state, reward, terminated, truncated, info

    def train(self, steps: int, rng_key: PRNGKey, verbose=1):
        """
        Args :
            steps : int
                Number of training steps to perform, where each step consists of interaactions and a policy update.
        """
        train_start_time = time.time()
        ac = self.ac

        rng_key, reset_rng_key = jax.random.split(rng_key, 2)
        if self.jax_env:
            env_obs, env_states, _ = self.env_reset(reset_rng_key)
        else:
            env_obs, _ = self.env.reset()
            env_states = None

        ep_lens, ep_rets, dones = (
            np.zeros(self.cfg.num_envs),
            np.zeros(self.cfg.num_envs),
            np.zeros(self.cfg.num_envs),
        )
        train_step_env_state = TrainStepEnvState(
            ep_lens=ep_lens,
            ep_rets=ep_rets,
            dones=dones,
            env_obs=env_obs,
            env_states=env_states,
            total_env_steps=self.total_env_steps,
            training_steps=self.training_steps,
        )

        start_step = self.total_env_steps
        if verbose:
            pbar = tqdm(total=steps + self.total_env_steps, initial=start_step)

        env_rollout_size = self.cfg.steps_per_env * self.cfg.num_envs

        while self.total_env_steps < steps:
            rng_key, train_rng_key = jax.random.split(rng_key, 2)
            train_step_env_state, train_step_metrics = self.train_step(train_rng_key, train_step_env_state)
            # TODO: once we have a TrainState we don't need this
            self.total_env_steps = train_step_env_state.total_env_steps
            self.training_steps = train_step_env_state.training_steps

            # evaluate the current trained actor periodically
            if (
                self.eval_loop is not None
                and tools.reached_freq(self.total_env_steps, self.cfg.eval_freq, step_size=env_rollout_size)
                and self.total_env_steps > self.cfg.num_seed_steps
            ):
                rng_key, eval_rng_key = jax.random.split(rng_key, 2)
                eval_results = self.evaluate(
                    eval_rng_key,
                    num_envs=self.cfg.num_eval_envs,
                    steps_per_env=self.cfg.eval_steps,
                    eval_loop=self.eval_loop,
                    params=ac.actor,
                    apply_fn=ac.act,
                )
                self.logger.store(
                    tag="test",
                    ep_ret=eval_results["eval_ep_rets"],
                    ep_len=eval_results["eval_ep_lens"],
                    append=False,
                )
                self.logger.log(self.total_env_steps)
                self.logger.reset()

            # log training metrics
            self.logger.store(tag="train", **train_step_metrics.train, append=False)
            self.logger.store(tag="train_stats", **train_step_metrics.train_stats, append=False)
            self.logger.store(tag="time", **train_step_metrics.time, append=False)

            if verbose:
                pbar.update(n=env_rollout_size)

            # log time information
            total_time = time.time() - train_start_time
            if tools.reached_freq(self.total_env_steps, self.cfg.log_freq):
                self.logger.store(
                    tag="time",
                    append=False,
                    total=total_time,
                    SPS=self.total_env_steps / total_time,
                    total_env_steps=self.total_env_steps,
                )
            self.logger.log(self.total_env_steps)
            self.logger.reset()

            # save checkpoints
            if tools.reached_freq(self.total_env_steps, self.cfg.save_freq):
                self.save(os.path.join(self.logger.model_path, f"ckpt_{self.total_env_steps}.jx"))

    def train_step(
        self, rng_key: PRNGKey, train_step_env_state: TrainStepEnvState
    ) -> Tuple[TrainStepEnvState, TrainStepMetrics]:
        """
        Perform a single training step

        In SAC this is composed of collecting cfg.steps_per_env * cfg.num_envs of interaction data with a random sample or policy (depending on cfg.num_seed_steps)
        then performing gradient updates

        TODO: If a jax-env is used, this step is jitted
        """
        ac = self.ac

        env_obs = train_step_env_state.env_obs
        env_states = train_step_env_state.env_states
        ep_lens = train_step_env_state.ep_lens
        ep_rets = train_step_env_state.ep_rets
        total_env_steps = train_step_env_state.total_env_steps
        training_steps = train_step_env_state.training_steps

        train_custom_stats = defaultdict(list)
        train_metrics = dict()
        train_metrics["ep_ret"] = []
        train_metrics["ep_len"] = []
        time_metrics = dict()

        # perform a rollout
        rollout_time_start = time.time()
        for _ in range(self.cfg.steps_per_env):
            rng_key, env_rng_key = jax.random.split(rng_key, 2)
            (actions, next_env_obs, next_env_states, rewards, terminations, truncations, infos,) = self._env_step(
                env_rng_key,
                env_obs,
                env_states,
                ac.actor,
                seed=total_env_steps <= self.cfg.num_seed_steps,
            )

            dones = terminations | truncations
            dones = np.array(dones)
            rewards = np.array(rewards)
            # TODO: handle dict observations + is there a more memory efficient way to store o_{t} and o_{t+1} without repeating a lot?
            true_next_env_obs = next_env_obs.copy()
            ep_lens += 1
            ep_rets += rewards

            masks = ((~dones) | (truncations)).astype(float)
            if dones.any():
                # note for continuous task wrapped envs where there is no early done, all envs finish at the same time unless
                # they are staggered. So masks is never false.
                # if you want to always value bootstrap set masks to true.
                for i, d in enumerate(dones):
                    if d:
                        train_metrics["ep_ret"].append(ep_rets[i])
                        train_metrics["ep_len"].append(ep_lens[i])
                        true_next_env_obs[i] = infos["final_observation"][i]
                        final_info = infos["final_info"][i]
                        if "stats" in final_info:
                            for k in final_info["stats"]:
                                train_custom_stats[k].append(final_info["stats"][k])
                ep_lens[dones] = 0.0
                ep_rets[dones] = 0.0

            self.replay_buffer.store(
                env_obs=env_obs,
                reward=rewards,
                action=actions,
                mask=masks,
                next_env_obs=true_next_env_obs,
            )

            env_obs = next_env_obs
            env_states = next_env_states

        rollout_time = time.time() - rollout_time_start
        time_metrics["rollout_time"] = rollout_time

        # update policy
        if self.total_env_steps >= self.cfg.num_seed_steps:
            update_time_start = time.time()
            update_actor = training_steps % self.cfg.actor_update_freq == 0
            update_target = training_steps % self.cfg.target_update_freq == 0
            for _ in range(self.cfg.grad_updates_per_step):
                rng_key, update_rng_key, sample_key = jax.random.split(rng_key, 3)
                batch = self.replay_buffer.sample_random_batch(sample_key, self.cfg.batch_size)
                batch = TimeStep(**batch)
                (new_actor, new_critic, new_target_critic, new_temp, aux,) = self.update_parameters(
                    update_rng_key,
                    ac.actor,
                    ac.critic,
                    ac.target_critic,
                    ac.temp,
                    batch,
                    update_actor,
                    update_target,
                )
                ac.actor = new_actor
                ac.critic = new_critic
                ac.target_critic = new_target_critic
                ac.temp = new_temp
            update_time = time.time() - update_time_start
            critic_update_aux: loss.CriticUpdateAux = aux["critic_update_aux"]
            actor_update_aux: loss.ActorUpdateAux = aux["actor_update_aux"]
            temp_update_aux: loss.TempUpdateAux = aux["temp_update_aux"]
            train_metrics["critic_loss"] = float(critic_update_aux.critic_loss)
            train_metrics["q1"] = float(critic_update_aux.q1)
            train_metrics["q2"] = float(critic_update_aux.q2)
            train_metrics["temp"] = float(temp_update_aux.temp)
            if update_actor:
                train_metrics["actor_loss"] = float(actor_update_aux.actor_loss)
                train_metrics["entropy"] = float(actor_update_aux.entropy)
                train_metrics["target_entropy"] = float(self.cfg.target_entropy)
                if self.cfg.learnable_temp:
                    train_metrics["temp_loss"] = float(temp_update_aux.temp_loss)
            time_metrics["update_time"] = update_time

        train_step_env_state = train_step_env_state.replace(
            ep_lens=ep_lens,
            ep_rets=ep_rets,
            env_obs=env_obs,
            env_states=env_states,
            total_env_steps=total_env_steps + self.cfg.num_envs * self.cfg.steps_per_env,
            training_steps=training_steps + 1,
            dones=dones,
        )

        return train_step_env_state, TrainStepMetrics(time=time_metrics, train=train_metrics, train_stats=train_custom_stats)

    @partial(jax.jit, static_argnames=["self", "update_actor", "update_target"])
    def update_parameters(
        self,
        rng_key: PRNGKey,
        actor: Model,
        critic: Model,
        target_critic: Model,
        temp: Model,
        batch: TimeStep,
        update_actor: bool,
        update_target: bool,
    ):
        rng_key, critic_update_rng_key = jax.random.split(rng_key, 2)
        new_critic, critic_update_aux = loss.update_critic(
            critic_update_rng_key,
            actor,
            critic,
            target_critic,
            temp,
            batch,
            self.cfg.discount,
            self.cfg.backup_entropy,
        )
        # init dummy values
        new_actor, actor_update_aux = actor, loss.ActorUpdateAux()
        new_temp, temp_update_aux = temp, loss.TempUpdateAux(temp=temp())
        new_target = target_critic

        if update_target:
            new_target = loss.update_target(critic, target_critic, self.cfg.tau)
        if update_actor:
            rng_key, actor_update_rng_key = jax.random.split(rng_key, 2)
            new_actor, actor_update_aux = loss.update_actor(actor_update_rng_key, actor, critic, temp, batch)
            if self.cfg.learnable_temp:
                new_temp, temp_update_aux = loss.update_temp(temp, actor_update_aux.entropy, self.cfg.target_entropy)
        return (
            new_actor,
            new_critic,
            new_target,
            new_temp,
            dict(
                critic_update_aux=critic_update_aux,
                actor_update_aux=actor_update_aux,
                temp_update_aux=temp_update_aux,
            ),
        )

    def state_dict(self):
        # TODO add option to save buffer?
        state_dict = dict(
            ac=self.ac.state_dict(),
            total_env_steps=self.total_env_steps,
            training_steps=self.training_steps,
            logger=self.logger.state_dict(),
        )
        return state_dict

    def save(self, save_path: str):
        state_dict = self.state_dict()
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(state_dict))

    def load(self, data):
        self.ac = self.ac.load(data["ac"])
        self.total_env_steps = data["total_env_steps"]
        self.training_steps = data["training_steps"]
        self.logger.load(data["logger"])
        return self
