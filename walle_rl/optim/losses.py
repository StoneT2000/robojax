
import functools

from typing import Optional, Union
import chex
import jax
import jax.numpy as jnp

Scalar = chex.Scalar
Array = chex.Array

def l2_loss(preds: Array, targets: Array) -> Array:
    """
    L2 loss function
    """
    chex.assert_type([preds, targets], float)
    return ((preds - targets) ** 2) / 2