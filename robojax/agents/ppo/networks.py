import os
from functools import partial
from pathlib import Path
from typing import Any, Tuple

import distrax
import flax
import flax.linen as nn
import flax.serialization
import flax.struct as struct
import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey

from robojax.models.model import Model

Params = flax.core.FrozenDict[str, Any]


def mlp(sizes, activation, output_activation=None):
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

    def _log_prob_from_distribution(self, dist: distrax.Distribution, act: Array) -> Array:
        return dist.log_prob(act)

    def __call__(self, x) -> distrax.Distribution:
        a = self.actor(x)
        dist = self.explorer(a)
        return dist, a


@struct.dataclass
class StepAux:
    value: Array
    log_p: Array


@struct.dataclass
class ActorCritic:
    """
    ActorCritic model. Manages the actor and critic models
    """

    actor: Model
    critic: Model

    @classmethod
    def create(
        cls,
        rng_key: PRNGKey,
        actor: nn.Module,
        critic: nn.Module,
        explorer: nn.Module,
        sample_obs,
        sample_acts,
        actor_optim: optax.GradientTransformation = optax.adam(3e-4),
        critic_optim: optax.GradientTransformation = optax.adam(3e-4),
    ) -> "ActorCritic":
        rng_key, actor_rng_key, critic_rng_key, temp_rng_key = jax.random.split(rng_key, 4)

        actor = Actor(actor=actor, explorer=explorer)
        actor = Model.create(model=actor, key=actor_rng_key, sample_input=sample_obs, tx=actor_optim)

        critic = Model.create(model=critic, key=critic_rng_key, sample_input=sample_obs, tx=critic_optim)
        return cls(actor=actor, critic=critic)

    @partial(jax.jit)
    def step(self, rng_key: PRNGKey, ac: "ActorCritic", obs) -> Tuple[Array, StepAux]:
        dist, _ = ac.actor(obs)
        dist: distrax.Distribution
        a = dist.sample(seed=rng_key)
        log_p = dist.log_prob(a)
        v = ac.critic(obs)
        v = jnp.squeeze(v, -1)
        return a, StepAux(value=v, log_p=log_p)

    @partial(jax.jit, static_argnames=["deterministic"])
    def act(self, rng_key: PRNGKey, actor: Actor, obs, deterministic=False):
        if deterministic:
            _, a = actor(obs)
            return a
        dist, _ = actor(obs)
        return dist.sample(seed=rng_key)

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
        )

    def save(self, save_path: str):
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.state_dict()))

    def load(self, params_dict: Params):
        self.actor = self.actor.load_state_dict(params_dict["actor"])
        self.critic = self.critic.load_state_dict(params_dict["critic"])
        return self

    def load_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            params_dict = flax.serialization.from_bytes(self.state_dict(), f.read())
        self.load(params_dict)
        return self
