"""Proximal policy optimization training.
See: https://arxiv.org/pdf/1707.06347.pdf
"""

from dataclasses import dataclass
import functools
import time
from typing import Callable, Dict, Optional
from chex import ArrayTree, PRNGKey

import distrax
import gym
import numpy as np
from walle_rl.architecture.model import Model
from walle_rl.common.random import PRNGSequence
import jax
from walle_rl.agents.ppo.buffer import PPOBuffer, Batch
from walle_rl.architecture.ac.core import Actor, ActorCritic, Params
from walle_rl.agents.base import Policy
from walle_rl.buffer.buffer import GenericBuffer
from walle_rl.common.rollout import Rollout
from walle_rl.logger.logger import Logger
from walle_rl.optim.pg import clipped_surrogate_pg_loss
import jax.numpy as jnp
import chex

@functools.partial(jax.jit, static_argnames=["gamma", "gae_lambda"])
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, gamma: float, gae_lambda: float):
    N = len(rewards)
    advantages = jnp.zeros((N + 1))
    # advantages = []
    gae = 0.0
    not_dones = ~dones
    
    value_diffs = gamma * values[1:] * not_dones - values[:-1]
    deltas = rewards + value_diffs

    # TODO vectorize for loop below
    def body_fun(gae, t):
        gae = deltas[t] + gamma * gae_lambda * not_dones[t] * gae
        return gae, gae
    indices = jnp.arange(N)[::-1]
    gae, advantages = jax.lax.scan(body_fun, 0.0, indices,)
    
    advantages = advantages[::-1]
    return jax.lax.stop_gradient(advantages)

@dataclass
class PPO(Policy):
    """
    Runs the PPO algorithm.
    """

    ### Mostly just to conveniently store the hyperparameters and also to generally enforce these
    ### parameters to not be changed over time by the user as they're usually initialized at the start and only changed by the algo
    ### Otherwise mostly written in a functional manner

    max_ep_len: int
    gamma: Optional[float] = 0.99
    gae_lambda: Optional[float] = 0.97
    clip_ratio: Optional[float] = 0.2
    ent_coef: Optional[float] = 0.0
    pi_coef: Optional[float] = 1.0
    vf_coef: Optional[float] = 1.0
    dapg_lambda: Optional[float] = 0.1
    dapg_damping: Optional[float] = 0.99
    target_kl: Optional[float] = 0.01
    
    def __hash__(self) -> int:
        # TODO neat trick to tell jax this object is hashable and can be made constant
        # This is so we can define jittable methods as member functions in a OOP style, although this should be avoided
        return id(self)

    def train_loop(
        self,
        rng: PRNGSequence,
        ac: ActorCritic,
        buffer: PPOBuffer,
        env: gym.Env,
        steps_per_epoch: int,
        rollout_callback: Callable = None,
        logger: Logger = None,
        verbose=1,
        batch_size=2048,
        update_iters=80,
        start_epoch=0,
        n_epochs=100,
        critic_warmup_epochs=0,
        train_callback: Callable = None,
    ):
        # simple wrapped training loop function
        for epoch in range(start_epoch, start_epoch + n_epochs):
            update_actor = start_epoch >= critic_warmup_epochs
            update_critic = True
            self.train_step(
                rng=rng,
                ac=ac,
                buffer=buffer,
                env=env,
                steps_per_epoch=steps_per_epoch,
                rollout_callback=rollout_callback,
                logger=logger,
                verbose=verbose,
                batch_size=batch_size,
                update_iters=update_iters,
                update_actor=update_actor,
                update_critic=update_critic,
            )
            logger.store("train", epoch=epoch, append=False)
            logger.store("train", env_interactions=steps_per_epoch * buffer.n_envs * (epoch + 1), append=False)
            if train_callback is not None:
                early_stop = train_callback(epoch=epoch)
                if early_stop is not None and early_stop:
                    break

    def train_step(
        self,
        rng: PRNGSequence,
        ac: ActorCritic,
        buffer: PPOBuffer,
        env: gym.Env,
        steps_per_epoch: int,
        rollout_callback: Callable,
        # start_epoch=0,
        # n_epochs=100,
        logger: Logger = None,
        verbose=1,
        batch_size=2048,
        update_iters=80,
        update_actor=True,
        update_critic=True,
    ):
        rollout = Rollout()

        def policy(o):
            res = ac.step(key=next(rng), obs=o)
            res["actions"] = np.array(res["actions"])
            return res

        buffer.reset()

        def wrapped_rollout_cb(observations, next_observations, pi_output, actions, rewards, infos, dones, timeouts):
            buffer.store(
                obs_buf=observations,
                act_buf=actions,
                rew_buf=rewards,
                val_buf=pi_output["val"],
                logp_buf=pi_output["logp_a"],
                done_buf=dones,
            )
            if logger is not None:
                logger.store(tag="train", v_vals=pi_output["val"])
            if rollout_callback is not None:
                return rollout_callback(
                    observations, next_observations, pi_output, actions, rewards, infos, dones, timeouts
                )

        # ac.eval()

        # TODO - add jax rollout collection method for gyms like brax and gymnax
        rollout.collect(
            policy=policy,
            env=env,
            n_envs=buffer.n_envs,
            steps=steps_per_epoch+1,
            rollout_callback=wrapped_rollout_cb,
            max_ep_len=self.max_ep_len,
            logger=logger,
            verbose=verbose,
        )
        # ac.train()
        update_start_time = time.time_ns()
        advantages = gae_advantages(
            buffer.buffers["rew_buf"][:-1],
            buffer.buffers["done_buf"][:-1],
            buffer.buffers["val_buf"],
            self.gamma,
            self.gae_lambda,
        )

        buffer.buffers["adv_buf"] = advantages
        returns = advantages + buffer.buffers["val_buf"][-1, :]  # TODO check device?
        buffer.buffers["ret_buf"] = returns
        # apply normalization trick
        buffer.buffers["adv_buf"] = (buffer.buffers["adv_buf"] - buffer.buffers["adv_buf"].mean()) / (
            buffer.buffers["adv_buf"].std() + 1e-8
        )
        buffer.buffer_to_jax()
        for update_iter in range(update_iters):
            batch = buffer.sample_batch(key=next(rng), batch_size=batch_size, drop_last_batch=True)
            # TODO - add grad accumulation,
            res = PPO.update_parameters_step(
                actor=ac.actor,
                critic=ac.critic,
                clip_ratio=self.clip_ratio,
                update_actor=update_actor,
                update_critic=update_critic,
                batch=batch,
            )

            ac.actor = res["new_actor"]
            ac.critic = res["new_critic"]
            info_a, info_c = res["info_a"], res["info_c"]
            logger.store(
                "train", critic_loss=info_c["critic_loss"]
            )
            if info_a is not None:
                logger.store("train", actor_loss=info_a["loss_pi"], entropy=info_a["entropy"])

        update_end_time = time.time_ns()
        logger.store("train", update_time=(update_end_time - update_start_time) * 1e-9, append=False)

        # TODO add dapg and make it jittable
        # if dapg:
        #     logger.store("train", dapg_lambda=self.dapg_lambda, append=False)
        #     if update_actor:
        #         self.dapg_lambda *= self.dapg_damping

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["clip_ratio", "update_actor", "update_critic"])
    def update_parameters_step(
        actor: Model, critic: Model, clip_ratio: float, update_actor: bool, update_critic: bool, batch: Batch
    ):
        info_a, info_c = None, None
        new_actor = actor
        new_critic = critic
        if update_actor:
            grads_a_fn = jax.grad(
                PPO.actor_loss_fn(clip_ratio=clip_ratio, actor_apply_fn=actor.apply_fn, batch=batch), has_aux=True
            )
            grads, info_a = grads_a_fn(actor.params)
            new_actor = actor.apply_gradient(grads=grads)
        if update_critic:
            grads_c_fn = jax.grad(PPO.critic_loss_fn(critic_apply_fn=critic.apply_fn, batch=batch), has_aux=True)
            grads, info_c = grads_c_fn(critic.params)
            new_critic = critic.apply_gradient(grads=grads)

        return dict(new_actor=new_actor, new_critic=new_critic, info_a=info_a, info_c=info_c)

    @staticmethod
    def actor_loss_fn(clip_ratio: float, actor_apply_fn: Callable, batch: Batch):
        def loss_fn(actor_params: Params):
            obs, act, adv, logp_old = batch.obs_buf, batch.act_buf, batch.adv_buf, batch.logp_buf
            # ac.pi.val()
            dist, _ = actor_apply_fn(actor_params, obs)
            dist: distrax.Distribution
            logp = dist.log_prob(act)
            # ac.pi.train()
            
            ratio = jnp.exp(logp - logp_old)
            clip_adv = jax.lax.clamp(1.0 - clip_ratio, ratio, 1.0 + clip_ratio) * adv
            loss_pi = -jnp.mean(jnp.minimum(ratio * adv, clip_adv), axis=0)
            entropy = dist.entropy().mean()

            info = dict(loss_pi=loss_pi, entropy=entropy, logp_old=logp_old.mean(), clip_adv=clip_adv)
            return loss_pi, info

        return loss_fn

    @staticmethod
    def critic_loss_fn(critic_apply_fn: Model, batch: Batch):
        def loss_fn(critic_params: Params):
            obs, ret = batch.obs_buf, batch.ret_buf
            v = critic_apply_fn(critic_params, obs)
            v = jnp.squeeze(v, -1)
            critic_loss = jnp.mean(jnp.square(v - ret), axis=0)
            return critic_loss, dict(critic_loss=critic_loss)

        return loss_fn
