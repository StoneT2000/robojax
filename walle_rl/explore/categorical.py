import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from walle_rl.architecture.model import Model

Array = chex.Array
Scalar = chex.Scalar


class Categorical(nn.Module):
    categorical = True

    def __call__(self, a) -> distrax.Distribution:
        dist = distrax.Categorical(logits=a)
        return dist

    def _log_prob_from_distribution(
        self, dist: distrax.Distribution, a: Array
    ) -> Array:
        return dist.log_prob(a)
