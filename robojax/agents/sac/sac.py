import os
import time
from dis import disco
from functools import partial
from pathlib import Path
from typing import Any, Callable, Tuple

import distrax
import flax
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey
from tqdm import tqdm

from robojax.agents.base import BasePolicy
from robojax.agents.sac import loss
from robojax.agents.sac.config import SACConfig, TimeStep
from robojax.agents.sac.networks import ActorCritic, DiagGaussianActor
from robojax.data import buffer
from robojax.data.buffer import GenericBuffer
from robojax.data.loop import EnvAction, EnvObs, GymLoop, JaxLoop
from robojax.models import Model
from robojax.models.model import Params


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

        assert self.cfg.max_episode_length is not None

        self.step = 0
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
        self.replay_buffer = GenericBuffer(
            buffer_size=self.cfg.replay_buffer_capacity,
            n_envs=self.cfg.num_envs,
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
            next_env_obs, next_env_state, reward, terminated, truncated, info = self.env_step(env_rng_key, env_state, a)
        else:
            a = np.asarray(a)
            next_env_obs, reward, terminated, truncated, info = self.env.step(a)
            done = np.logical_or(terminated, truncated)
            next_env_state = None
        return a, next_env_obs, next_env_state, reward, done, info

    def train(self, rng_key: PRNGKey, verbose=1):
        train_start_time = time.time()
        ac = self.ac
        episodes = 0
        ep_lens, ep_rets, dones = (
            np.zeros(self.cfg.num_envs),
            np.zeros(self.cfg.num_envs),
            np.zeros(self.cfg.num_envs),
        )
        rng_key, reset_rng_key = jax.random.split(rng_key, 2)
        if self.jax_env:
            env_obs, env_states = self.env_reset(reset_rng_key)
        else:
            env_obs = self.env.reset()
            env_states = None

        if verbose:
            pbar = tqdm(total=self.cfg.num_train_steps, initial=self.step)
        while self.step < self.cfg.num_train_steps:
            # evaluate the current trained actor periodically
            if (
                self.eval_loop is not None
                and self.step % self.cfg.eval_freq == 0
                and self.step > 0
                and self.step >= self.cfg.num_seed_steps
                and self.cfg.eval_freq > 0
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

            # perform a rollout (usually single step in all parallel envs)
            rng_key, env_rng_key = jax.random.split(rng_key, 2)
            (
                actions,
                next_env_obs,
                next_env_states,
                rewards,
                dones,
                infos,
            ) = self._env_step(
                env_rng_key,
                env_obs,
                env_states,
                ac.actor,
                seed=self.step < self.cfg.num_seed_steps,
            )
            dones = np.array(dones)
            rewards = np.array(rewards)
            true_next_env_obs = next_env_obs.copy()
            ep_lens += 1
            ep_rets += rewards
            self.step += 1

            masks = ((~dones) | (ep_lens == self.cfg.max_episode_length)).astype(float)
            if dones.any():
                self.logger.store(
                    tag="train",
                    ep_ret=ep_rets[dones],
                    ep_len=ep_lens[dones],
                    append=False,
                )
                self.logger.log(self.total_env_steps)
                self.logger.reset()
                episodes += dones.sum()
                ep_lens[dones] = 0.0
                ep_rets[dones] = 0.0
                for i, d in enumerate(dones):
                    if d:
                        true_next_env_obs[i] = infos[i]["terminal_observation"]

            self.replay_buffer.store(
                env_obs=env_obs,
                reward=rewards,
                action=actions,
                mask=masks,
                next_env_obs=true_next_env_obs,
            )

            env_obs = next_env_obs
            env_states = next_env_states

            # update policy
            if self.step >= self.cfg.num_seed_steps:
                update_time_start = time.time()
                rng_key, update_rng_key, sample_key = jax.random.split(rng_key, 3)
                update_actor = self.step % self.cfg.actor_update_freq == 0
                update_target = self.step % self.cfg.target_update_freq == 0
                for _ in range(self.cfg.grad_updates_per_step):
                    batch = self.replay_buffer.sample_random_batch(sample_key, self.cfg.batch_size)
                    batch = TimeStep(**batch)
                    (
                        new_actor,
                        new_critic,
                        new_target_critic,
                        new_temp,
                        aux,
                    ) = self.update_parameters(
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
                if self.cfg.log_freq > 0 and self.step % self.cfg.log_freq == 0:
                    self.logger.store(
                        tag="train",
                        append=False,
                        critic_loss=float(critic_update_aux.critic_loss),
                        q1=float(critic_update_aux.q1),
                        q2=float(critic_update_aux.q2),
                        temp=float(temp_update_aux.temp),
                    )
                    if update_actor:
                        self.logger.store(
                            tag="train",
                            actor_loss=float(actor_update_aux.actor_loss),
                            entropy=float(actor_update_aux.entropy),
                            target_entropy=float(self.cfg.target_entropy),
                            append=False,
                        )
                        if self.cfg.learnable_temp:
                            self.logger.store(
                                tag="train",
                                temp_loss=float(temp_update_aux.temp_loss),
                                append=False,
                            )
                    self.logger.store(tag="time", append=False, update_time=update_time)
                    self.logger.log(self.total_env_steps)
                    self.logger.reset()
            if verbose:
                pbar.update(n=1)

            total_time = time.time() - train_start_time
            if self.cfg.log_freq > 0 and self.step % self.cfg.log_freq == 0:
                self.logger.store(
                    tag="time",
                    append=False,
                    total=total_time,
                    step=self.step,
                )
                self.logger.log(self.total_env_steps)
                self.logger.reset()

            if self.step % self.cfg.save_freq == 0 and self.step >= self.cfg.num_seed_steps:
                self.save(os.path.join(self.logger.model_path, f"ckpt_{self.total_env_steps}.jx"))

    @property
    def total_env_steps(self):
        return self.step * self.cfg.num_envs * self.cfg.steps_per_env

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
        state_dict = dict(ac=self.ac.state_dict(), step=self.step, logger=self.logger.state_dict())
        return state_dict

    def save(self, save_path: str):
        state_dict = self.state_dict()
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(state_dict))

    def load(self, data):
        self.ac = self.ac.load(data["ac"])
        self.step = data["step"]
        self.logger.load(data["logger"])
        return self
