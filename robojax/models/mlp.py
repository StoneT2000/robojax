"""MLP class"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp

from .types import NetworkConfig


@dataclass
class MLPArchConfig:
    features: List[int]
    activation: Union[Callable, str] = "relu"
    output_activation: Union[Callable, str] = None


@dataclass
class MLPConfig(NetworkConfig):
    type = "mlp"
    arch_cfg: MLPArchConfig


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    """
    Parameters
    ----------
    features - hidden units in each layer

    activation - internal activation

    output_activation - activation after final layer, default is None
    """

    features: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = None
    final_ortho_scale: float = jnp.sqrt(2)

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat, kernel_init=default_init())(x))
        x = nn.Dense(self.features[-1], kernel_init=default_init(self.final_ortho_scale))(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
