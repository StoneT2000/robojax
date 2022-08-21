import time
from dis import disco
from functools import partial
from typing import Callable, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey
from flax import struct

from robojax.agents.base import BasePolicy
from robojax.agents.sac.config import SACConfig, TimeStep
from robojax.agents.sac.networks import ActorCritic
from robojax.data import buffer
from robojax.data.buffer import GenericBuffer
from robojax.data.loop import EnvAction, EnvObs
from robojax.logger.logger import Logger
from robojax.models import Model
from robojax.models.model import Params
from robojax.utils.spaces import get_action_dim, get_obs_shape


@struct.dataclass
class CriticUpdateAux:
    critic_loss: Array
    q1: Array
    q2: Array


@struct.dataclass
class ActorUpdateAux:
    actor_loss: Array = None
    entropy: Array = None


@struct.dataclass
class TempUpdateAux:
    temp_loss: Array = None
    temp: Array = None


class SAC(BasePolicy):
    def __init__(
        self,
        jax_env: bool,
        seed_sampler: Callable[[PRNGKey], EnvAction],
        observation_space,
        action_space,
        env=None,
        env_step=None,
        env_reset=None,
        cfg: SACConfig = {},
    ):
        super().__init__(jax_env, env, env_step, env_reset)
        self.cfg = SACConfig(**cfg)

        self.step = 0
        self.seed_sampler = seed_sampler

        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        buffer_config = dict(
            action=((self.action_dim,), action_space.dtype),
            reward=((), np.float32),
            done=((), np.bool8),
        )
        if isinstance(self.obs_shape, dict):
            buffer_config["env_obs"] = (
                self.obs_shape,
                {k: self.observation_space[k].dtype for k in self.observation_space},
            )
        else:
            buffer_config["env_obs"] = (self.obs_shape, np.float32)
        buffer_config["next_env_obs"] = buffer["env_obs"]
        self.replay_buffer = GenericBuffer(
            self.cfg.replay_buffer_capacity, n_envs=1, config=buffer_config
        )
        if self.cfg.target_entropy is None:
            self.cfg.target_entropy = -self.action_dim / 2

    def train(self, rng_key: PRNGKey, ac: ActorCritic, logger: Logger, verbose=1):
        stime = time.time()
        episodes = 0
        ep_len, ep_ret, done = 0, 0, True
        rng_key, reset_rng_key = jax.random.split(rng_key, 2)
        env_obs, env_state = self.env_reset(reset_rng_key)
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    if self.step % self.cfg.eval_freq == 0:
                        # evaluate
                        pass
                rng_key, reset_rng_key = jax.random.split(rng_key, 2)
                env_obs, env_state = self.env_reset(reset_rng_key)
                ep_len, ep_ret = 0
                episodes += 1

            rng_key, act_rng_key, env_rng_key = jax.random.split(rng_key, 3)
            if self.step < self.cfg.num_seed_steps:
                a = self.seed_sampler(act_rng_key)
            else:
                a = ac.act(act_rng_key, actor=ac.actor, obs=env_obs, deterministic=True)

            # update policy
            if self.step >= self.cfg.num_seed_steps:
                rng_key, update_rng_key, sample_key = jax.random.split(rng_key, 3)
                update_actor = self.step % self.cfg.actor_update_freq == 0
                update_target = self.step % self.cfg.target_update_freq == 0
                batch = self.replay_buffer.sample_random_batch(
                    sample_key, self.batch_size
                )
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
                critic_update_aux: CriticUpdateAux = aux["critic_update_aux"]
                actor_update_aux: ActorUpdateAux = aux["actor_update_aux"]
                temp_update_aux: TempUpdateAux = aux["temp_update_aux"]

                logger.store(
                    tag="train",
                    critic_loss=critic_update_aux.critic_loss,
                    temp=temp_update_aux.temp,
                )
                if update_actor:
                    logger.store(
                        tag="train",
                        actor_loss=actor_update_aux.actor_loss,
                        entropy=actor_update_aux.entropy,
                    )
                    if self.cfg.learnable_temp:
                        logger.store(tag="train", temp_loss=temp_update_aux.temp_loss)

            next_env_obs, next_env_state, reward, done, info = self.env_step(
                env_rng_key, env_state, a
            )

            ep_len += 1
            ep_ret += reward
            self.step += 1

            mask = 0.0
            if not done or ep_len == self.cfg.max_episode_length:
                mask = 1.0
            else:
                # 0 here means we don't use the q value of the next state and action.
                # we bootstrap whenever we have a time limit termination
                mask = 0.0
            self.replay_buffer.store(
                env_obs=env_obs,
                reward=reward,
                action=a,
                mask=mask,
                next_env_obs=next_env_obs,
            )

            env_obs = next_env_obs
            env_state = next_env_state

    @partial(jax.jit, static_argnames=["self"])
    def update_target(self, critic: Model, target_critic: Model) -> Model:
        new_target_critic_params = jax.tree_util.tree_map(
            lambda cp, tcp: cp * self.cfg.tau + tcp * (1 - self.cfg.tau),
            critic.params,
            target_critic.params,
        )
        return target_critic.replace(params=new_target_critic_params)

    @partial(jax.jit, static_argnames=["self"])
    def update_critic(
        self,
        rng_key: PRNGKey,
        actor: Model,
        critic: Model,
        target_critic: Model,
        temp: Model,
        batch: TimeStep,
    ) -> Tuple[Model, CriticUpdateAux]:
        dist: distrax.Distribution = actor(batch.next_env_obs)
        next_actions = dist.sample(seed=rng_key)
        next_log_probs = dist.log_prob(next_actions)
        next_q1, next_q2 = target_critic(batch.next_env_obs, next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        target_q = batch.reward + self.cfg.discount * batch.mask * next_q

        if self.cfg.backup_entropy:
            target_q -= self.cfg.discount * batch.mask * temp() * next_log_probs

        def critic_loss_fn(critic_params: Params):
            q1, q2 = critic.apply_fn(critic_params, batch.env_obs, batch.action)
            critic_loss = (jnp.square(q1 - target_q) + jnp.square(q2 - target_q)).mean()
            return critic_loss, CriticUpdateAux(critic_loss=critic_loss, q1=q1, q2=q2)

        grad_fn = jax.grad(critic_loss_fn, has_aux=True)
        grads, aux = grad_fn(critic.params)
        new_critic = critic.apply_gradients(grads=grads)
        return new_critic, aux

    @partial(jax.jit, static_argnames=["self"])
    def update_actor(
        self,
        rng_key: PRNGKey,
        actor: Model,
        critic: Model,
        temp: Model,
        batch: TimeStep,
    ) -> Tuple[Model, ActorUpdateAux]:
        def actor_loss_fn(actor_params: Params):
            dist: distrax.Distribution = actor.apply_fn(actor_params, batch.env_obs)
            actions = dist.sample(seed=rng_key)
            log_probs = dist.log_prob(actions)
            a_q1, a_q2 = critic(batch.env_obs, batch.action)
            a_q = jnp.minimum(a_q1, a_q2)
            actor_loss = (temp() * log_probs - a_q).mean()
            return actor_loss, ActorUpdateAux(
                actor_loss=actor_loss, entropy=-log_probs.mean()
            )

        grad_fn = jax.grad(actor_loss_fn, has_aux=True)
        grads, aux = grad_fn(actor.params)
        new_actor = actor.apply_gradients(grads=grads)
        return new_actor, aux

    @partial(jax.jit, static_argnames=["self"])
    def update_temp(self, temp: Model, entropy: float) -> Tuple[Model, TempUpdateAux]:
        def temp_loss_fn(temp_params: Params):
            temp_val = temp.apply_fn(temp_params)
            temp_loss = temp_val * (entropy - self.cfg.target_entropy).mean()
            return temp_loss, TempUpdateAux(temp_loss=temp_loss, temp=temp_val)

        grad_fn = jax.grad(temp_loss_fn, has_aux=True)
        grads, aux = grad_fn(temp.params)
        new_temp = temp.apply_gradients(grads=grads)
        return new_temp, aux

    @partial(jax.jit, static_argnames=["self"])
    def update_parameters(
        self,
        rng_key: PRNGKey,
        actor: Model,
        critic: Model,
        target_critic: Model,
        temp: Model,
        batch: TimeStep,
        update_actor,
        update_target,
    ):
        rng_key, critic_update_rng_key = jax.random.split(rng_key, 2)
        new_critic, critic_update_aux = self.update_critic(
            critic_update_rng_key, actor, critic, target_critic, temp, batch
        )
        new_actor, actor_update_aux = actor, ActorUpdateAux()
        new_temp, temp_update_aux = temp, TempUpdateAux(temp=temp())
        new_target = target_critic
        if update_actor:
            rng_key, actor_update_rng_key = jax.random.split(rng_key, 2)
            new_actor, actor_update_aux = self.update_actor(
                actor_update_rng_key, actor, critic, temp, batch
            )
            if self.cfg.learnable_temp:
                new_temp, temp_update_aux = self.update_temp(
                    temp, actor_update_aux.entropy
                )
        if update_target:
            new_target = self.update_target(critic, target_critic)

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
