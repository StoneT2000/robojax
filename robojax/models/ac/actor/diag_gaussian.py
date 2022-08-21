from typing import Callable, Sequence, Tuple
import flax.linen as nn
import distrax
import jax.numpy as jnp

from robojax.models.ac.core import mlp

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