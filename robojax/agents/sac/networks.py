import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey
from flax import struct
from tensorflow_probability.substrates import jax as tfp

from robojax.models import MLP, Model
from robojax.models.model import Params

tfd = tfp.distributions
tfb = tfp.bijectors


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
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_critics,
        )
        qs = VmapCritic(self.features, self.activation)(obs, acts)
        return qs


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class DiagGaussianActor(nn.Module):
    features: Sequence[int]
    act_dims: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    tanh_squash_distribution: bool = True

    state_dependent_std: bool = True
    log_std_range: Tuple[float, float] = (-5.0, 2.0)

    def setup(self) -> None:
        if self.state_dependent_std:
            # Add final dense layer initialization scale and orthogonal init
            self.log_std = nn.Dense(self.act_dims, kernel_init=default_init(1))
        else:
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.act_dims,))
        self.mlp = MLP(self.features, self.activation, self.output_activation)

        # scale of orthgonal initialization is recommended to be (high - low) / 2.
        # We always assume envs are normalized so 1 is correct
        self.action_head = nn.Dense(self.act_dims, kernel_init=default_init(1))

    def __call__(self, x, deterministic=False):
        x = self.mlp(x)
        a = self.action_head(x)
        if not self.tanh_squash_distribution:
            a = nn.tanh(a)
        if deterministic:
            return nn.tanh(a)
        if self.state_dependent_std:
            log_std = self.log_std(x)

            # Spinning up implementaation
            log_std = nn.tanh(log_std)
            log_std = self.log_std_range[0] + 0.5 * (self.log_std_range[1] - self.log_std_range[0]) * (log_std + 1)
        else:
            log_std = self.log_std
        # log_std = jnp.clip(log_std, self.log_std_range[0], self.log_std_range[1])
        dist = tfd.MultivariateNormalDiag(a, jnp.exp(log_std))
        # distrax has some numerical imprecision bug atm where calling sample then log_prob can raise NaNs. tfd is more stable at the moment
        # dist = distrax.MultivariateNormalDiag(a, jnp.exp(log_std))
        if self.tanh_squash_distribution:
            # dist = distrax.Transformed(distribution=dist, bijector=distrax.Block(distrax.Tanh(), ndims=1))
            dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())
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


@struct.dataclass
class ActorCritic:
    actor: Model
    critic: Model
    target_critic: Model
    temp: Model

    @classmethod
    def create(
        cls,
        rng_key: PRNGKey,
        sample_obs: Array,
        sample_acts: Array,
        actor: DiagGaussianActor = None,
        critic: DoubleCritic = None,
        actor_optim: optax.GradientTransformation = optax.adam(3e-4),
        critic_optim: optax.GradientTransformation = optax.adam(3e-4),
        initial_temperature: float = 1.0,
        temperature_optim: optax.GradientTransformation = optax.adam(3e-4),
    ) -> "ActorCritic":
        rng_key, actor_rng_key, critic_rng_key, temp_rng_key = jax.random.split(rng_key, 4)

        if actor is None:
            actor = DiagGaussianActor(features=[256, 256], act_dims=sample_acts.shape[-1])
        actor = Model.create(actor, actor_rng_key, sample_obs, actor_optim)
        if critic is None:
            critic = DoubleCritic(features=[256, 256], num_critics=2)
        critic = Model.create(critic, critic_rng_key, [sample_obs, sample_acts], critic_optim)

        target_critic = Model.create(critic, critic_rng_key, [sample_obs, sample_acts])

        temp = Model.create(Temperature(initial_temperature), temp_rng_key, tx=temperature_optim)

        return cls(actor=actor, critic=critic, target_critic=target_critic, temp=temp)

    @partial(jax.jit)
    def act(self, rng_key: PRNGKey, actor: DiagGaussianActor, obs):
        return actor(obs, deterministic=True), {}

    @partial(jax.jit)
    def sample(self, rng_key: PRNGKey, actor: DiagGaussianActor, obs):
        return actor(obs).sample(seed=rng_key), {}

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            target_critic=self.target_critic.state_dict(),
            temp=self.temp.state_dict(),
        )

    def save(self, save_path: str):
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.state_dict()))

    def load(self, params_dict: Params):
        self.actor = self.actor.load_state_dict(params_dict["actor"])
        self.critic = self.critic.load_state_dict(params_dict["critic"])
        self.target_critic = self.target_critic.load_state_dict(params_dict["target_critic"])
        self.temp = self.temp.load_state_dict(params_dict["temp"])
        return self

    def load_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            params_dict = flax.serialization.from_bytes(self.state_dict(), f.read())
        self.load(params_dict)
        return self
