from functools import partial
from typing import Callable, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey

from robojax.models import MLP, Model
from robojax.models.ac.core import mlp


class Critic(nn.Module):
    # TODO make a decorator that injects these network parameters based on some config
    features: Sequence[int]
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, obs: Array, acts: Array) -> Array:
        x = jnp.concatenate([obs, acts], -1)
        critic = MLP((*self.features, 1), self.activation)(x)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    features: Sequence[int]
    activation: Callable[[Array], Array] = nn.relu
    num_critics: int = 2

    @nn.compact
    def __call__(self, obs: Array, acts: Array):
        VmapCritic = nn.vmap(
            Critic,
            variable_axes=0,
            split_rngs=True,
            in_axes=None,
            out_axes=0,
            axis_size=self.num_critics,
        )
        qs = VmapCritic(features=self.features, activation=self.activation)(obs, acts)
        return qs


class DiagGaussianActor(nn.Module):

    features: Sequence[int]
    act_dims: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = None

    tanh_squash_distribution: bool = True

    log_std_scale: float = -0.5
    state_dependent_std: bool = False
    log_std_range: Tuple[float, float] = [-10.0, 2.0]

    def setup(self) -> None:
        if self.state_dependent_std:
            # Add final dense layer initialization scale and orthogonal init
            self.log_std = nn.Dense(self.act_dims)
        else:
            self.log_std = self.param(
                "log_std",
                lambda rng, act_dims, log_std_scale: jnp.ones(act_dims) * log_std_scale,
                self.act_dims,
                self.log_std_scale,
            )
        self.mlp = mlp(self.features, self.activation, self.output_activation)
        self.action_head = nn.Dense(self.act_dims)

    def __call__(self, x):
        x = self.mlp(x)
        if self.state_dependent_std:
            log_std = self.log_std(x)
            log_std = jnp.clip(log_std, self.log_std_range[0], self.log_std_range[1])
        else:
            log_std = self.log_std
        a = self.action_head(x)
        if not self.tanh_squash_distribution:
            a = nn.tanh(a)
        dist = distrax.MultivariateNormalDiag(a, jnp.exp(log_std))
        if self.tanh_squash_distribution:
            dist = distrax.Transformed(distribution=dist, bijector=distrax.Tanh())
        return dist


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self):
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


class ActorCritic:
    actor: Model
    critic: Model
    target_critic: Model

    def __init__(
        self,
        rng_key: PRNGKey,
        sample_obs: Array,
        sample_acts: Array,
        actor: DiagGaussianActor = None,
        critic: DoubleCritic = None,
        actor_optim: optax.GradientTransformation = optax.adam(3e-4),
        critic_optim: optax.GradientTransformation = optax.adam(3e-4),
        initial_temperature: float = 1.0,
        temperature_optim: optax.GradientTransformation = optax.adam(3e-4),
    ) -> None:
        rng_key, actor_rng_key, critic_rng_key, temp_rng_key = jax.random.split(
            rng_key, 4
        )

        if actor is None:
            actor = DiagGaussianActor(
                features=[256, 256], act_dims=sample_acts.shape[-1]
            )
        self.actor = Model.create(actor, actor_rng_key, sample_obs, actor_optim)
        if critic is None:
            critic = DoubleCritic(features=[256, 256], num_critics=2)
        self.critic = Model.create(
            critic, critic_rng_key, [sample_obs, sample_acts], critic_optim
        )

        self.target_critic = Model.create(
            critic, critic_rng_key, [sample_obs, sample_acts]
        )

        self.temp = Model.create(
            Temperature(initial_temperature), temp_rng_key, tx=temperature_optim
        )

    @partial(jax.jit, static_argnames=["self", "deterministic"])
    def act(self, rng_key: PRNGKey, actor: DiagGaussianActor, obs, deterministic=False):
        if deterministic:
            return actor(obs)
        dist, _ = actor(obs)
        return dist.sample(seed=rng_key)
