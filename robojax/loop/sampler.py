from dataclasses import dataclass, fields
from functools import partial

import jax
from chex import PRNGKey


class BufferSampler:
    def __init__(self, buffer, buffer_size: int, num_envs: int) -> None:
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.buffer_keys = fields(self.buffer)

    @partial(jax.jit, static_argnames=["self", "batch_size", "drop_last_batch"])
    def sample_batch(
        self, rng_key: PRNGKey, batch_size: int, drop_last_batch: bool = True
    ):
        pass

    @partial(jax.jit, static_argnames=["self", "batch_size"])
    def sample_random_batch(self, rng_key: PRNGKey, batch_size: int):
        """
        Sample a batch of data with replacement
        """
        rng_key, batch_ids_rng_key = jax.random.split(rng_key)
        batch_ids = jax.random.randint(
            batch_ids_rng_key, shape=(batch_size,), minval=0, maxval=self.buffer_size
        )
        # batch_ids = (np.random.randint(0, self.buffer_size, size=batch_size) + self.ptr) % self.buffer_size
        rng_key, env_ids_rng_key = jax.random.split(rng_key)
        env_ids = jax.random.randint(
            env_ids_rng_key, shape=(batch_size,), minval=0, maxval=self.num_envs
        )

        return self._get_batch_by_ids(batch_ids=batch_ids, env_ids=env_ids)

    @partial(jax.jit, static_argnames=["self"])
    def _get_batch_by_ids(self, batch_ids, env_ids):
        """
        statefully retrieve batch of data
        """
        data = dict()
        for field in self.buffer_keys:
            data[field.name] = getattr(self.buffer, field.name)[batch_ids, env_ids]
        return data
