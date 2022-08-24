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
from robojax.agents.sac.networks import ActorCritic, DiagGaussianActor
from robojax.data import buffer
from robojax.data.buffer import GenericBuffer
from robojax.data.loop import EnvAction, EnvObs, GymLoop, JaxLoop
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
        if isinstance(cfg, dict):
            self.cfg = SACConfig(**cfg)
        else:
            self.cfg = cfg

        self.step = 0
        self.seed_sampler = seed_sampler

        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        buffer_config = dict(
            action=((self.action_dim,), action_space.dtype),
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
            buffer_size=self.cfg.replay_buffer_capacity, n_envs=1, config=buffer_config
        )
        if self.cfg.target_entropy is None:
            self.cfg.target_entropy = -self.action_dim / 2

        if self.jax_env:
            self._env_step = jax.jit(self._env_step, static_argnames=["seed"])
            self.eval_loop = JaxLoop(
                self.env_reset,
                self.env_step,
                reset_env=True,
            )
        else:
            self.eval_loop = GymLoop(self.env)

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
            next_env_obs, next_env_state, reward, done, info = self.env_step(
                env_rng_key, env_state, a
            )
        else:
            # print(a.shape)
            # import ipdb;ipdb.set_trace()
            next_env_obs, reward, done, info = self.env.step(np.asarray(a))
            next_env_state = None
        return a, next_env_obs, next_env_state, reward, done, info

    
    def policy(self, r, actor, obs):
        return actor(obs, deterministic=True), {}


    def train(self, rng_key: PRNGKey, ac: ActorCritic, logger: Logger, verbose=1):
        stime = time.time()
        episodes = 0
        ep_len, ep_ret, done = 0, 0, False
        rng_key, reset_rng_key = jax.random.split(rng_key, 2)
        if self.jax_env:
            env_obs, env_state = self.env_reset(reset_rng_key)
        else:
            env_obs = self.env.reset()
            env_state = None
        from tqdm import tqdm
        pbar=tqdm(total=self.cfg.num_train_steps)
        while self.step < self.cfg.num_train_steps:
            # print(self.step, self.cfg.eval_freq)
            if self.step > 0 and self.step > self.cfg.num_seed_steps:
                if self.step % self.cfg.eval_freq == 0:
                    # evaluate
                    num_eval_envs = 1
                    rng_key, *eval_rng_keys = jax.random.split(rng_key, num_eval_envs + 1)      
                    eval_buffer, _ = self.eval_loop.rollout(rng_keys=jnp.stack(eval_rng_keys),
                        params=ac.actor,
                        apply_fn=self.policy,
                        # apply_fn = lambda rng, obs : (ac.actor(obs, deterministic=True), {}),
                        steps_per_env=1000,
                    )
                    ep_lens = np.asarray(eval_buffer['ep_len'])
                    ep_rets = np.asarray(eval_buffer['ep_ret'])
                    # import ipdb;ipdb.set_trace()
                    # ep_rews = np.asarray(eval_buffer.reward)
                    episode_ends = np.asarray(eval_buffer['done'])
                    ep_rets = ep_rets[episode_ends].flatten()
                    ep_lens = ep_lens[episode_ends].flatten()
                    # print("EVAL", self.step, dict(avg_ret=ep_rets.mean(), avg_len=ep_lens.mean()))
                    logger.store(
                        tag="test",
                        ep_ret=ep_rets,
                        ep_len=ep_lens,
                        append=False,
                    )
            if done:
                
                rng_key, reset_rng_key = jax.random.split(rng_key, 2)
                if self.jax_env:
                    env_obs, env_state = self.env_reset(reset_rng_key)
                else:
                    env_obs = self.env.reset()
                # print("===",ep_ret)
                # pbar.set_postfix(dict(ep_ret=ep_ret, ep_len=ep_len))
                ep_len, ep_ret = 0, 0
                episodes += 1

            
            rng_key, env_rng_key = jax.random.split(rng_key, 2)
            a, next_env_obs, next_env_state, reward, done, info = self._env_step(env_rng_key, env_obs, env_state, ac.actor, seed=self.step < self.cfg.num_seed_steps)

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

            # update policy
            if self.step >= self.cfg.num_seed_steps:
                rng_key, update_rng_key, sample_key = jax.random.split(rng_key, 3)
                update_actor = self.step % self.cfg.actor_update_freq == 0
                update_target = self.step % self.cfg.target_update_freq == 0
                batch = self.replay_buffer.sample_random_batch(
                    sample_key, self.cfg.batch_size
                )
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
                critic_update_aux: CriticUpdateAux = aux["critic_update_aux"]
                actor_update_aux: ActorUpdateAux = aux["actor_update_aux"]
                temp_update_aux: TempUpdateAux = aux["temp_update_aux"]
                if self.step % self.cfg.log_freq == 0:
                    logger.store(
                        tag="train",
                        critic_loss=critic_update_aux.critic_loss,
                        q1=critic_update_aux.q1,
                        q2=critic_update_aux.q2,
                        temp=temp_update_aux.temp,
                    )
                    if update_actor:
                        logger.store(
                            tag="train",
                            actor_loss=actor_update_aux.actor_loss,
                            entropy=actor_update_aux.entropy,
                            target_entropy=self.cfg.target_entropy
                        )
                        if self.cfg.learnable_temp:
                            logger.store(tag="train", temp_loss=temp_update_aux.temp_loss)
                    stats = logger.log(self.step)
                    logger.reset()
            pbar.update(n=1)
            

    @partial(jax.jit, static_argnames=["self"])
    def update_target(self, critic: Model, target_critic: Model) -> Model:
        """
        update targret_critic with polyak averaging
        """
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
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, CriticUpdateAux(critic_loss=critic_loss, q1=q1.mean(), q2=q2.mean())

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
            q1, q2 = critic(batch.env_obs, actions)
            q = jnp.minimum(q1, q2)
            actor_loss = (temp() * log_probs - q).mean()
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
