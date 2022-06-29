from dataclasses import dataclass
import time
from typing import Callable, Dict, Optional
from chex import PRNGKey

import distrax
import gym
from walle_rl.common.random import PRNGSequence
import jax
from walle_rl.agents.ppo.buffer import PPOBuffer, Batch
from walle_rl.architecture.ac.core import ActorCritic
from walle_rl.agents.base import Policy
from walle_rl.buffer.buffer import GenericBuffer
from walle_rl.common.rollout import Rollout
from walle_rl.logger.logger import Logger
from walle_rl.optim.pg import clipped_surrogate_pg_loss
import jax.numpy as jnp


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
            return ac.step(key=next(rng), obs=o)

        buffer.reset()
        
        def wrapped_rollout_cb(observations, next_observations, pi_output, actions, rewards, infos, dones, timeouts):
            observations, next_observations, pi_output, actions, rewards, infos, dones, timeouts
            buffer.store(
                obs_buf=observations,
                act_buf=actions,
                rew_buf=rewards,
                val_buf=pi_output["val"],
                logp_buf=pi_output["logp_a"],
                done_buf=dones,
            )
            if logger is not None: logger.store(tag="train", v_vals=pi_output["val"])
            if rollout_callback is not None:
                return rollout_callback(observations, next_observations, pi_output, actions, rewards, infos, dones, timeouts)

        # ac.eval()
        rollout.collect(
            policy=policy,
            env=env,
            n_envs=buffer.n_envs,
            buf=buffer,
            steps=steps_per_epoch,
            rollout_callback=wrapped_rollout_cb,
            max_ep_len=self.max_ep_len,
            logger=logger,
            verbose=verbose,
        )
        # ac.train()
        # logger.store("train", RolloutTime=rollout_delta_time, critic_warmup_epochs=critic_warmup_epochs, append=False)
        update_start_time = time.time_ns()
        # advantage normalization before update
        # update_pi = epoch >= critic_warmup_epochs
        # update(update_pi = update_pi)
        for update_iter in range(update_iters):
            batch = buffer.sample_batch(batch_size=batch_size, drop_last_batch=True)
            # info = self.gradient(ac=ac, batch=batch)
            if update_actor:
                info_a = self.get_actor_loss_fn(ac)
                grads_a_fn = jax.grad(info_a["loss_pi_fn"])
                ac.actor.apply_gradient(grads_a_fn)
                logger.store("train", entropy=info_a["entropy"])
            if update_critic:
                info_c = self.get_critic_loss_fn(ac)
                grads_c_fn = jax.grad(info_c["loss_v_fn"])
                ac.critic.apply_gradient(grads_c_fn)

        update_end_time = time.time_ns()
        logger.store("train", update_time=(update_end_time - update_start_time) * 1e-9, append=False)
        # if dapg:
        #     logger.store("train", dapg_lambda=self.dapg_lambda, append=False)
        #     if update_actor:
        #         self.dapg_lambda *= self.dapg_damping

    def get_actor_loss_fn(self, ac: ActorCritic, batch: Batch):
        obs, act, adv, logp_old = batch["obs_buf"], batch["act_buf"], batch["adv_buf"], batch["logp_buf"]

        # ac.pi.´val()
        dist, a = ac.actor(obs)
        logp = ac.actor._log_prob_from_distribution(dist, a)
        # ac.pi.train()
        prob_ratios = jnp.exp(logp - logp_old)
        loss_pi_fn = clipped_surrogate_pg_loss(prob_ratios_t=prob_ratios, adv_t=adv, clip_ratio=self.clip_ratio)

        entropy = dist.entropy()

        info = dict(loss_pi_fn=loss_pi_fn, entropy=entropy)
        return info

    def get_critic_loss_fn(self, ac: ActorCritic, batch: Batch):
        obs, ret = batch["obs_buf"], batch["ret_buf"]
        loss_v_fn = ((ac.critic(obs) - ret) ** 2).mean()

        return dict(loss_v_fn=loss_v_fn)
