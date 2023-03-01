"""
Loss functions for the PPO agent
"""

from functools import partial
from typing import Callable

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import struct

from robojax.agents.ppo.config import TimeStep
from robojax.models import Model, Params


@struct.dataclass
class ActorAux:
    actor_loss: chex.Array = 0.0
    entropy: chex.Array = 0.0
    approx_kl: chex.Array = 0.0


@struct.dataclass
class CriticAux:
    critic_loss: chex.Array = 0.0


def actor_loss_fn(clip_ratio: float, entropy_coef: float, actor_apply_fn: Callable, batch: TimeStep):
    def loss_fn(actor_params: Params):
        obs, act, adv, logp_old = batch.env_obs, batch.action, batch.adv, batch.log_p
        # TODO: turn into eval mode here?
        # ac.pi.val()
        dist, _ = actor_apply_fn(actor_params, obs)
        dist: distrax.Distribution
        logp = dist.log_prob(act)
        # ac.pi.train()
        log_r = logp - logp_old
        ratio = jnp.exp(log_r)
        clip_adv = jax.lax.clamp(1.0 - clip_ratio, ratio, 1.0 + clip_ratio) * adv
        actor_loss = -jnp.mean(jnp.minimum(ratio * adv, clip_adv), axis=0)
        entropy = dist.entropy().mean()
        entropy_loss = -entropy * entropy_coef

        approx_kl = (ratio - 1 - log_r).mean()

        total_loss = actor_loss + entropy_loss
        info = ActorAux(
            actor_loss=actor_loss,
            entropy=entropy,
            approx_kl=jax.lax.stop_gradient(approx_kl),
        )
        return total_loss, info

    return loss_fn


def critic_loss_fn(critic_apply_fn: Callable, batch: TimeStep):
    def loss_fn(critic_params: Params):
        obs, ep_ret = batch.env_obs, batch.ep_ret
        v = critic_apply_fn(critic_params, obs)
        v = jnp.squeeze(v, -1)
        critic_loss = jnp.mean(jnp.square(v - ep_ret), axis=0)
        return critic_loss, CriticAux(critic_loss=critic_loss)

    return loss_fn
