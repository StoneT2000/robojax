"""
Loss functions for the PPO agent
"""

from typing import Callable

import distrax
import jax
import jax.numpy as jnp

from robojax.agents.ppo.config import TimeStep
from robojax.models import Model, Params


def actor_loss_fn(clip_ratio: float, actor_apply_fn: Callable, batch: TimeStep):
    def loss_fn(actor_params: Params):
        obs, act, adv, logp_old = batch.env_obs, batch.action, batch.adv, batch.log_p
        # ac.pi.val()
        dist, _ = actor_apply_fn(actor_params, obs)
        dist: distrax.Distribution
        logp = dist.log_prob(act)
        # ac.pi.train()

        ratio = jnp.exp(logp - logp_old)
        clip_adv = jax.lax.clamp(1.0 - clip_ratio, ratio, 1.0 + clip_ratio) * adv
        loss_pi = -jnp.mean(jnp.minimum(ratio * adv, clip_adv), axis=0)
        entropy = dist.entropy().mean()

        info = dict(
            loss_pi=loss_pi,
            entropy=entropy,
            logp_old=logp_old.mean(),
            clip_adv=clip_adv,
        )
        return loss_pi, info

    return loss_fn


def critic_loss_fn(critic_apply_fn: Model, batch: TimeStep):
    def loss_fn(critic_params: Params):
        obs, ret = batch.env_obs, batch.ret
        v = critic_apply_fn(critic_params, obs)
        v = jnp.squeeze(v, -1)
        critic_loss = jnp.mean(jnp.square(v - ret), axis=0)
        return critic_loss, dict(critic_loss=critic_loss)

    return loss_fn
