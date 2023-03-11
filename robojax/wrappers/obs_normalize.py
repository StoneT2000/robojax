from functools import partial
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RunningMeanStd:
    shape: Tuple
    dtype: Any
    mean: chex.Array
    var: chex.Array
    count: chex.Array = 1e-4

    @staticmethod
    def init(shape, dtype):
        return RunningMeanStd(shape, dtype, jnp.zeros(shape, dtype), jnp.ones(shape, dtype=dtype))

    @partial(jax.jit)
    def update(self, batch: chex.Array):
        batch_mean = jnp.mean(batch, axis=0)
        batch_var = jnp.var(batch, axis=0)
        batch_count = batch.shape[0]
        return self.update_from_moments(batch_mean, batch_var, batch_count)

    @partial(jax.jit)
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        return self.replace(mean=new_mean, var=new_var, count=new_count)


@struct.dataclass
class Normalize:
    obs_rms: RunningMeanStd
    clip_obs: float = 10.0

    @staticmethod
    def init(shape, dtype):
        return Normalize(RunningMeanStd.init(shape, dtype))

    @partial(jax.jit)
    def update(self, batch: chex.Array):
        return self.replace(obs_rms=self.obs_rms.update(batch))

    def normalize_obs(self, batch):
        return jnp.clip(
            (batch - self.obs_rms.mean) / jnp.sqrt(self.obs_rms.var + 1e-8),
            -self.clip_obs,
            self.clip_obs,
        )
