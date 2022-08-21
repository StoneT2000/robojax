import time
from functools import partial
from typing import Any, Callable, Dict, Tuple

import chex
import distrax
import gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from robojax.agents.base import BasePolicy

from robojax.agents.ppo.config import PPOConfig, TimeStep
from robojax.agents.ppo.loss import ActorAux, CriticAux, actor_loss_fn, critic_loss_fn
from robojax.data.loop import GymLoop, JaxLoop, RolloutAux
from robojax.data.sampler import BufferSampler
from robojax.logger.logger import Logger
from robojax.models.ac.core import ActorCritic
from robojax.models.model import Model, Params

PRNGKey = chex.PRNGKey


@partial(jax.jit, static_argnames=["gamma", "gae_lambda"])
@partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(rewards, dones, values, gamma: float, gae_lambda: float):
    N = len(rewards)
    # values is of shape (N+1,)
    advantages = jnp.zeros((N + 1))
    not_dones = ~dones

    value_diffs = gamma * values[1:] * not_dones - values[:-1]
    # in value_diffs we zero out whenever an episode was finished.
    # steps where done = True, then values[1:] is zeroed as it is the value for the next episode
    deltas = rewards + value_diffs

    def body_fun(gae, t):
        gae = deltas[t] + gamma * gae_lambda * not_dones[t] * gae
        return gae, gae

    indices = jnp.arange(N)[::-1] # N - 1, N - 2, ..., 0
    _, advantages = jax.lax.scan(
        body_fun,
        0.0,
        indices,
    )

    advantages = advantages[::-1]
    return jax.lax.stop_gradient(advantages)


class PPO(BasePolicy):
    def __init__(
        self,
        jax_env: bool,
        env=None,
        env_step=None,
        env_reset=None,
        cfg: PPOConfig = {},
        
    ) -> None:
        self.jax_env = jax_env
        self.cfg = PPOConfig(**cfg)

        self.last_env_obs_states = None
        if self.jax_env:

            def rollout_callback(
                action, env_obs, reward, ep_ret, ep_len, next_env_obs, done, info, aux
            ):
                return TimeStep(
                    action=action,
                    env_obs=env_obs,
                    reward=reward,
                    adv=0,
                    log_p=aux["log_p"],
                    ep_ret=ep_ret,
                    value=aux["value"],
                    done=done,
                    ep_len=ep_len,
                    info=info,
                )

            self.loop = JaxLoop(env_reset, env_step, rollout_callback=rollout_callback, reset_env=self.cfg.reset_env)
            self.last_env_states = None
            self.collect_buffer = jax.jit(
                self.collect_buffer,
                static_argnames=["rollout_steps_per_env", "num_envs", "apply_fn"],
            )
        else:
            # we expect env to be a vectorized env now
            def rollout_callback(
                action, env_obs, reward, ep_ret, ep_len, next_env_obs, done, info, aux
            ):
                batch_size = len(env_obs)
                return dict(
                    action=action,
                    env_obs=env_obs,
                    reward=reward,
                    adv=jnp.zeros((batch_size)),
                    log_p=aux["log_p"],
                    ep_ret=ep_ret,
                    value=aux["value"],
                    done=done,
                    ep_len=jnp.array(ep_len),
                )

            self.loop = GymLoop(env, rollout_callback=rollout_callback)

    def train(
        self,
        update_iters: int,
        steps_per_epoch: int,
        num_envs: int,
        rng_key: PRNGKey,
        epochs: int,
        ac: ActorCritic,
        batch_size: int,
        logger: Logger,
        train_callback: Callable = None,
        verbose: int = 1,
    ):
        # TODO create full jittable version
        train_start_time = time.time()
        if self.jax_env:

            def apply_fn(rng_key, params, obs):
                actor, critic = params
                res = ac.step(rng_key, actor, critic, obs)
                return res

        else:

            def apply_fn(rng_key, params, obs):
                actor, critic = params
                res = ac.step(rng_key, actor, critic, obs)
                return np.array(res[0]), res[1]

        env_steps_per_epoch = num_envs * steps_per_epoch
        for t in range(epochs):
            rng_key, train_rng_key = jax.random.split(rng_key)
            actor, critic, aux = self.train_step(
                rng_key=train_rng_key,
                update_iters=update_iters,
                rollout_steps_per_env=steps_per_epoch,
                num_envs=num_envs,
                actor=ac.actor,
                critic=ac.critic,
                apply_fn=apply_fn,
                batch_size=batch_size,
            )
            ac.actor = actor
            ac.critic = critic

            buffer = aux["buffer"]
            ep_lens = np.asarray(buffer.ep_len)
            ep_rets = np.asarray(buffer.orig_ret)
            ep_rews = np.asarray(buffer.reward)
            episode_ends = np.asarray(buffer.done)

            if train_callback is not None:
                rng_key, train_callback_rng_key = jax.random.split(rng_key)
                train_callback(epoch=t, ac=ac, rng_key=train_callback_rng_key)

            if logger is not None:
                total_env_steps = (t + 1) * env_steps_per_epoch
                actor_loss_aux: ActorAux = aux["update_aux"]["actor_loss_aux"]
                critic_loss_aux: CriticAux = aux["update_aux"]["critic_loss_aux"]
                total_time = time.time() - train_start_time
                if episode_ends.any():
                    logger.store(
                        tag="train",
                        append=False,
                        ep_ret=ep_rets[episode_ends].flatten(),
                        ep_len=ep_lens[episode_ends].flatten(),
                    )
                logger.store(
                    tag="train",
                    append=False,
                    ep_rew=ep_rews.flatten(),
                    fps=env_steps_per_epoch / aux["rollout_time"],
                    env_steps=total_env_steps,
                    entropy=np.asarray(actor_loss_aux.entropy),
                    actor_loss=np.asarray(actor_loss_aux.actor_loss),
                    approx_kl=np.asarray(actor_loss_aux.approx_kl),
                    critic_loss=np.asarray(critic_loss_aux.critic_loss),
                    actor_updates=aux["update_aux"]["actor_updates"].item(),
                    critic_updates=aux["update_aux"]["critic_updates"].item()
                )

                logger.store(
                    tag="time",
                    append=False,
                    rollout=aux["rollout_time"],
                    update=aux["update_time"],
                    total=total_time,
                    sps=total_env_steps / total_time,
                    epoch=t,
                )
                stats = logger.log(total_env_steps)
                if verbose > 0:
                    if verbose == 1:
                        filtered_stat_keys = [
                            "train/ep_len_avg",
                            "train/ep_ret_avg",
                            "time/rollout",
                            "time/update",
                            "time/total",
                            "time/sps",
                            "time/epoch",
                            "test/ep_ret_avg",
                            "test/ep_len_avg",
                        ]
                        filtered_stats = {
                            k: stats[k] for k in filtered_stat_keys if k in stats
                        }
                        logger.pretty_print_table(filtered_stats)
                    else:
                        logger.pretty_print_table(stats)
                logger.reset()
    
    def train_step(
        self,
        rng_key: PRNGKey,
        update_iters: int,
        rollout_steps_per_env: int,
        num_envs: int,
        actor: Model,
        critic: Model,  # steps per env
        apply_fn: Callable,  # rollout function
        batch_size: int,
    ):
        rng_key, buffer_rng_key = jax.random.split(rng_key)
        rollout_s_time = time.time()
        
        # TODO can we prevent compilation here where init_env_obs_states=None the first time for reset_env=False?
        if not self.cfg.reset_env:
            if self.last_env_obs_states is None:
                rng_key, *env_reset_rng_keys = jax.random.split(rng_key, num_envs + 1)
                self.last_env_obs_states = jax.jit(jax.vmap(self.loop.env_reset))(jnp.stack(env_reset_rng_keys))
        buffer, info = self.collect_buffer(
            rng_key=buffer_rng_key,
            rollout_steps_per_env=rollout_steps_per_env,
            num_envs=num_envs,
            actor=actor,
            critic=critic,
            apply_fn=apply_fn,
            init_env_obs_states=self.last_env_obs_states
        )
        if not self.cfg.reset_env:
            info: RolloutAux
            self.last_env_obs_states = (info.final_env_obs, info.final_env_state) 

        rollout_time = time.time() - rollout_s_time
        update_s_time = time.time()
        actor, critic, update_aux = self.update_parameters(
            rng_key=rng_key,
            actor=actor,
            critic=critic,
            update_actor=True,
            update_critic=True,
            update_iters=update_iters,
            num_envs=num_envs,
            batch_size=batch_size,
            buffer=buffer,
        )
        
        # remove entries where we skipped updating actor due to early stopping
        actor_loss_aux: ActorAux = update_aux["actor_loss_aux"]
        actor_loss_aux.replace(
            approx_kl=actor_loss_aux.approx_kl
        )

        update_time = time.time() - update_s_time
        return (
            actor,
            critic,
            dict(
                buffer=buffer,
                update_aux=update_aux,
                rollout_time=rollout_time,
                update_time=update_time,
            ),
        )

    @partial(
        jax.jit,
        static_argnames=[
            "self",
            "update_actor",
            "update_critic",
            "update_iters",
            "num_envs",
            "batch_size",
        ],
    )
    def update_parameters(
        self,
        rng_key: PRNGKey,
        actor: Model,
        critic: Model,
        update_actor: bool,
        update_critic: bool,
        update_iters: int,
        num_envs: int,
        batch_size: int,
        buffer: TimeStep,
    ) -> Tuple[Model, Model, Dict]:
        """ Update the actor and critic parameters """
        sampler = BufferSampler(
            ["action", "env_obs", "log_p", "ep_ret", "adv"],
            buffer,
            buffer_size=buffer.action.shape[0],
            num_envs=num_envs,
        )

        def update_step_fn(data: Tuple[PRNGKey, Model, Model], unused):
            rng_key, actor, critic, update_actor, actor_updates, critic_updates = data
            rng_key, sample_rng_key = jax.random.split(rng_key)
            batch = TimeStep(**sampler.sample_random_batch(sample_rng_key, batch_size))
            info_a: ActorAux = None
            info_c = None
            new_actor = actor
            new_critic = critic


            def update_actor_fn(actor):
                grads_a_fn = jax.grad(
                    actor_loss_fn(
                        clip_ratio=self.cfg.clip_ratio,
                        entropy_coef=self.cfg.ent_coef,
                        actor_apply_fn=actor.apply_fn,
                        batch=batch,
                    ),
                    has_aux=True,
                )
                grads, info_a = grads_a_fn(actor.params)
                new_actor = actor.apply_gradients(grads=grads)

                return new_actor, info_a
            def skip_update_actor_fn(actor):
                return actor, ActorAux()
            new_actor, info_a = jax.lax.cond(update_actor, update_actor_fn, skip_update_actor_fn, actor)
            update_actor = jax.lax.cond(info_a.approx_kl > self.cfg.target_kl, lambda : False, lambda : True)
            actor_updates += 1 * update_actor
            
            if update_critic:
                grads_c_fn = jax.grad(
                    critic_loss_fn(critic_apply_fn=critic.apply_fn, batch=batch),
                    has_aux=True,
                )
                grads, info_c = grads_c_fn(critic.params)
                new_critic = critic.apply_gradients(grads=grads)
                critic_updates += 1
            return (rng_key, new_actor, new_critic, update_actor, actor_updates, critic_updates), dict(actor_loss_aux=info_a, critic_loss_aux=info_c)

        update_init = (rng_key, actor, critic, update_actor, 0, 0)
        carry, update_aux = jax.lax.scan(
            update_step_fn, update_init, (), length=update_iters
        )
        _, actor, critic, _, actor_updates, critic_updates = carry

        return actor, critic, dict(**update_aux, actor_updates=actor_updates, critic_updates=critic_updates)

    def collect_buffer(
        self,
        rng_key,
        rollout_steps_per_env: int,
        num_envs: int,
        actor,
        critic,
        apply_fn: Callable,
        init_env_obs_states = None,
    ):  
        # buffer collection is not jitted if env is not jittable
        
        # regardless this function returns a struct.dataclass object with all
        # the data in jax.numpy arrays for use
        rng_key, *env_rng_keys = jax.random.split(rng_key, num_envs + 1)
        buffer, aux = self.loop.rollout(
            env_rng_keys,
            params=(actor, critic),
            apply_fn=apply_fn,
            steps_per_env=rollout_steps_per_env + 1,  # extra 1 for final value computation
            max_episode_length=self.cfg.max_episode_length,
            init_env_obs_states=init_env_obs_states
        )
        buffer: TimeStep
        
        if not self.jax_env:
            # if not a jax based env, then buffer is a python dictionary and we
            # convert it
            buffer = TimeStep(**buffer)
        # rest of the code here is jitted and / or vmapped
        advantages = gae_advantages(
            buffer.reward[:-1],
            buffer.done[:-1],
            buffer.value,
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        returns = advantages + buffer.value[-1, :]
        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # TODO can we speed up this replace op?
        buffer = buffer.replace(
            adv=advantages,
            ep_ret=returns,
            reward=buffer.reward[:-1],
            orig_ret=buffer.ep_ret[:-1],
            log_p=buffer.log_p[:-1],
            action=buffer.action[:-1],
            env_obs=buffer.env_obs[:-1],
            done=buffer.done[:-1],
            ep_len=buffer.ep_len[:-1],
        )

        return buffer, aux
