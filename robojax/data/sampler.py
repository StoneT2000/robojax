"""
Samplers for data collected from environment loops
"""
from functools import partial
from typing import Any

import jax
from chex import PRNGKey
from flax import struct


@struct.dataclass
class BufferSampler2:
    # buffer_keys: struct.field(pytree_node=False)
    buffer: Any
    buffer_size: int  # once sampled beyond this size, data is shuffled
    curr_idx: int  # current index to sample from
    rng_key: PRNGKey

    @classmethod
    def create(
        cls,
        buffer,
        # buffer_keys,
        buffer_size: int,
        rng_key: PRNGKey,
    ):
        return cls(
            buffer=buffer,
            # buffer_keys=buffer_keys,
            buffer_size=buffer_size,
            curr_idx=0,
            rng_key=rng_key,
        )

    @partial(jax.jit, static_argnames=["batch_size"])
    def sample(self, batch_size: int):
        curr_idx = self.curr_idx
        rng_key, permute_key = jax.random.split(self.rng_key)

        def convert_data(x):
            x = jax.random.permutation(permute_key, x)
            return x

        buffer = jax.lax.cond(
            curr_idx == 0, lambda y: jax.tree_util.tree_map(lambda x: convert_data(x), y), lambda y: y, self.buffer
        )

        batch = jax.tree_util.tree_map(lambda x: x[curr_idx : curr_idx + batch_size], buffer)

        curr_idx = curr_idx + batch_size
        curr_idx = jax.numpy.where(curr_idx + batch_size < self.buffer_size, curr_idx, 0)
        return (
            self.replace(
                buffer=buffer,
                curr_idx=curr_idx,
                rng_key=rng_key,
            ),
            batch,
        )


class BufferSampler:
    """
    Samples batches of data from a given buffer. Expects buffer to be of type `flax.struct.dataclass`
    """

    def __init__(self, buffer_keys, buffer, buffer_size: int, num_envs: int) -> None:
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.buffer_keys = buffer_keys

    @partial(jax.jit, static_argnames=["self", "batch_size"])
    def sample_random_batch(self, rng_key: PRNGKey, batch_size: int):
        """
        Sample a batch of data with replacement
        """
        rng_key, batch_ids_rng_key = jax.random.split(rng_key)
        batch_ids = jax.random.randint(batch_ids_rng_key, shape=(batch_size,), minval=0, maxval=self.buffer_size)
        return self._get_batch_by_ids(batch_ids)

    @partial(jax.jit, static_argnames=["self"])
    def _get_batch_by_ids(self, ids):
        """
        retrieve batch of data via batch ids and env ids
        """
        data = {}
        for k in self.buffer_keys:
            data[k] = getattr(self.buffer, k)[ids]
        return data
