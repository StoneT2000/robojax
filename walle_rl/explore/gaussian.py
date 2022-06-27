"""
Exploration strategies. Code adopted from RLax https://github.com/deepmind/rlax/
"""

from typing import Optional

import chex
import jax
import jax.numpy as jnp

Array = chex.Array
Scalar = chex.Scalar


def add_gaussian_noise(
    key: Array,
    action: Array,
    stddev: float
) -> Array:
  """Returns continuous action with noise drawn from a Gaussian distribution.
  Args:
    key: a key from `jax.random`.
    action: continuous action scalar or vector.
    stddev: standard deviation of noise distribution.
  Returns:
    noisy action, of the same shape as input action.
  """
  chex.assert_type(action, float)

  noise = jax.random.normal(key, shape=action.shape) * stddev
  return action + noise