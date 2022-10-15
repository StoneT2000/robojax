from typing import Any, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax import struct

from robojax.agents.sac.config import TimeStep
from robojax.models import Model


@struct.dataclass
class CriticUpdateAux:
    critic_loss: Array
    q1: Array
    q2: Array


@struct.dataclass
class TempUpdateAux:
    temp_loss: Array = None
    temp: Array = None


def update_critic(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: TimeStep,
    discount: float,
    backup_entropy: bool,
) -> Tuple[Model, Any]:
    dist = actor(batch.next_env_obs)
    # next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_env_obs, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.reward + discount * batch.mask * next_q

    if backup_entropy:
        target_q -= discount * batch.mask * temp() * next_log_probs

    def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Any]:
        q1, q2 = critic.apply_fn(critic_params, batch.env_obs, batch.action)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, CriticUpdateAux(
            critic_loss=critic_loss, q1=q1.mean(), q2=q2.mean()
        )

    grad_fn = jax.grad(critic_loss_fn, has_aux=True)
    grads, aux = grad_fn(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, aux


@struct.dataclass
class ActorUpdateAux:
    actor_loss: Array = None
    entropy: Array = None


def update_actor(
    key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: TimeStep
) -> Tuple[Model, Any]:
    def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Any]:
        dist = actor.apply_fn(actor_params, batch.env_obs)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.env_obs, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, ActorUpdateAux(
            **{"actor_loss": actor_loss, "entropy": -log_probs.mean()}
        )

    grad_fn = jax.grad(actor_loss_fn, has_aux=True)
    grads, aux = grad_fn(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    return new_actor, aux


def update_temp(
    temp: Model, entropy: float, target_entropy: float
) -> Tuple[Model, Any]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn(temp_params)
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, TempUpdateAux(**{"temp": temperature, "temp_loss": temp_loss})

    grad_fn = jax.grad(temperature_loss_fn, has_aux=True)
    grads, aux = grad_fn(temp.params)
    new_temp = temp.apply_gradients(grads=grads)
    return new_temp, aux


def update_target(critic: Model, target_critic: Model, tau: float) -> Model:
    """
    update targret_critic with polyak averaging
    """
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)
