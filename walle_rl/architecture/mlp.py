from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """
    Initialize an MLP
    Parameters
    ----------
    features - hidden units in each layer

    activation - internal activation

    output_activation - activation after final layer, default is None
    """

    features: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = None

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
