# import torch.nn as nn
import functools
from typing import Any, Callable, Tuple, Union
from chex import Array
import distrax
import flax.struct as struct
import flax.linen as nn
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from walle_rl.architecture.model import Model
from walle_rl.common.random import PRNGSequence

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
    act_dims: int
    actor: nn.Module
    explorer: nn.Module
    log_std_scale: float = -0.5

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


@functools.partial(jax.jit, static_argnames=["actor_apply_fn", "critic_apply_fn"])
def _step(key, actor_apply_fn: Callable, actor_params: Params, critic_apply_fn: Callable, critic_params: Params, obs: np.ndarray):
    dist, _ = actor_apply_fn(actor_params, obs)
    a = dist.sample(seed=key)
    logp_a = dist.log_prob(a)
    v = critic_apply_fn(critic_params, obs)
    v = jnp.squeeze(
        v, -1
    )
    return dict(actions=a, val=v, logp_a=logp_a)


class ActorCritic:
    actor: Model
    critic: Model
    act_dims: int

    def __init__(
        self,
        rng: PRNGSequence,
        actor: nn.Module,
        critic: nn.Module,
        explorer,
        obs_shape,
        act_dims,
        actor_optim: optax.GradientTransformation,
        critic_optim: optax.GradientTransformation,
    ) -> None:
        self.act_dims = act_dims
        actor_module = Actor(act_dims=self.act_dims, actor=actor, explorer=explorer)
        self.actor = Model.create(model=actor_module, key=next(rng), input_shape=obs_shape, optimizer=actor_optim)
        self.critic = Model.create(model=critic, key=next(rng), input_shape=obs_shape, optimizer=critic_optim)

    def step(self, key, obs):
        res = _step(
            key=key,
            actor_apply_fn=self.actor.apply_fn,
            actor_params=self.actor.params,
            critic_apply_fn=self.critic.apply_fn,
            critic_params=self.critic.params,
            obs=obs,
        )
        return res
        # dist = self.actor(obs, method=self.actor._distribution)
        # a = dist.sample(seed=key)
        # logp_a = self.actor._log_prob_from_distribution(dist, a)
        # v = self.critic(obs)
        # return dict(actions=a, val=v, logp_a=logp_a)

    # @jax.jit
    def act(self, obs, key=None, deterministic=False):
        if deterministic:
            return self.actor(obs)
        dist: distrax.Distribution = self.actor(obs, method=self.actor.model._distribution)
        # TODO remove np array here
        return np.array(dist.sample(seed=key))
