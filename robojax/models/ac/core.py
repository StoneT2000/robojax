import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Tuple, Union

import distrax
import flax
import flax.linen as nn
import flax.serialization
import flax.struct as struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import Array, PRNGKey

from robojax.models.model import Model

Params = flax.core.FrozenDict[str, Any]


def count_vars(module):
    return sum([jnp.prod(p.shape) for p in module.parameters()])


def mlp(sizes, activation, output_activation=None):  # TODO
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Dense(sizes[j], sizes[j + 1])]
        if j < len(sizes) - 2:
            layers += [activation()]
        else:
            if output_activation is not None:
                layers += [output_activation()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    actor: nn.Module
    explorer: nn.Module

    def __hash__(self) -> int:
        return id(self)

    def _distribution(self, x):
        a = self.actor(x)
        dist = self.explorer(a)
        return dist

    def _log_prob_from_distribution(
        self, dist: distrax.Distribution, act: Array
    ) -> Array:
        return dist.log_prob(act)

    def __call__(self, x) -> distrax.Distribution:
        a = self.actor(x)
        dist = self.explorer(a)
        return dist, a


class ActorCritic:
    """
    ActorCritic model. Manages the actor and critic models
    """

    actor: Model
    critic: Model

    def __init__(
        self,
        rng_key: PRNGKey,
        actor: nn.Module,
        critic: nn.Module,
        explorer,
        sample_obs,
        act_dims,
        actor_optim: optax.GradientTransformation,
        critic_optim: optax.GradientTransformation,
    ) -> None:
        actor_module = Actor(actor=actor, explorer=explorer)
        rng_key, actor_rng_key = jax.random.split(rng_key)
        rng_key, critic_rng_key = jax.random.split(rng_key)
        self.actor = Model.create(
            model=actor_module,
            key=actor_rng_key,
            sample_input=sample_obs,
            tx=actor_optim,
        )
        self.critic = Model.create(
            model=critic, key=critic_rng_key, sample_input=sample_obs, tx=critic_optim
        )

    @partial(jax.jit, static_argnames=["self"])
    def step(self, rng_key: PRNGKey, actor: Model, critic: Model, obs):
        dist, _ = actor(obs)
        dist: distrax.Distribution
        a = dist.sample(seed=rng_key)
        log_p = dist.log_prob(a)
        v = critic(obs)
        v = jnp.squeeze(v, -1)
        return a, dict(value=v, log_p=log_p)

    @partial(jax.jit, static_argnames=["self", "deterministic"])
    def act(self, rng_key: PRNGKey, actor: Actor, obs, deterministic=False):
        if deterministic:
            return actor(obs)
        dist, _ = actor(obs)
        return dist.sample(seed=rng_key)

    def _state_dict(self):
        return dict(
            actor=self.actor._state_dict(),
            critic=self.critic._state_dict(),
        )

    def save(self, save_path: str):
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self._state_dict()))

    def load(self, load_path: str):
        with open(load_path, "rb") as f:
            params_dict = flax.serialization.from_bytes(self._state_dict(), f.read())
        self.actor = self.actor.replace(**params_dict["actor"])
        self.critic = self.critic.replace(**params_dict["critic"])
