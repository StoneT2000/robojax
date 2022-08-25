"""
Adapted from SB3
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from flax import struct
from gym import spaces


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: Max number of element in the buffer
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        device="cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size

        self.ptr = 0
        self.full = False
        self.n_envs = n_envs

    def store(self, *args, **kwargs) -> None:
        """
        store elements to the buffer.
        """
        raise NotImplementedError()

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.ptr

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.ptr = 0
        self.full = False


class GenericBuffer(BaseBuffer):
    """
    Generic buffer that stores key value items for vectorized environment outputs.

    Note that by default everything is stored as a numpy array in this buffer. See JaxBuffer for a version that stores with jax arrays.

    config - dict(k->v) where k is buffer name and v[0] is shape, v[1] is numpy dtype, v[2] is data is dict or not. if is_dict, then shape and dtype should be a dict of shapes and dtypes
    """

    def __init__(self, buffer_size: int, device="cpu", n_envs: int = 1, config=dict()):
        super().__init__(
            buffer_size=buffer_size,
            device=device,
            n_envs=n_envs,
        )
        self.is_dict = dict()
        self.config = config
        self.buffers = dict()

        self.ptr, self.path_start_idx, self.max_size = 0, [0] * n_envs, self.buffer_size

        self.batch_idx = None
        self.batch_inds = None
        self.batch_env_inds = None
        self.prepare_for_collection()

    def reset(self) -> None:
        self.prepare_for_collection()
        return super().reset()

    def prepare_for_collection(self):
        """
        setups up buffer as dictionary for each type of value being held.
        """
        for k in self.config.keys():
            shape, dtype = self.config[k]
            is_dict = False
            if isinstance(shape, dict):
                is_dict = True
            self.is_dict[k] = is_dict
            if is_dict:
                self.buffers[k] = dict()
                for part_key in shape.keys():
                    self.buffers[k][part_key] = np.zeros(
                        (self.buffer_size, self.n_envs) + shape[part_key],
                        dtype=dtype[part_key],
                    )
            else:
                self.buffers[k] = np.zeros(
                    (self.buffer_size, self.n_envs) + shape, dtype=dtype
                )

    def store(self, **kwargs):
        """
        store one timestep of agent-environment interaction to the buffer. If full, replaces the oldest entry
        """
        for k in kwargs.keys():
            data = kwargs[k]
            if self.is_dict[k]:
                for data_k in data.keys():
                    d = np.array(data[data_k]).copy()
                    d = d.reshape(self.buffers[k][data_k][self.ptr].shape)
                    self.buffers[k][data_k][self.ptr] = d
            else:
                d = np.array(data).copy()
                d = d.reshape(self.buffers[k][self.ptr].shape)
                self.buffers[k][self.ptr] = d
        self.ptr += 1
        if self.ptr == self.buffer_size:
            # wrap pointer around to start replacing items
            self.full = True
            self.ptr = 0

    # @partial(jax.jit, static_argnames=["self"])
    def _get_batch_by_ids(self, buffers, batch_ids, env_ids):
        """
        statefully retrieve batch of data
        """
        batch_data = dict()
        for k in buffers.keys():
            data = buffers[k]
            if self.is_dict[k]:
                batch_data[k] = dict()
                for data_k in data.keys():
                    batch_data[k][data_k] = data[data_k][batch_ids, env_ids]
            else:
                batch_data[k] = data[batch_ids, env_ids]
        return batch_data

    def _prepared_for_sampling(self, batch_size, drop_last_batch=True):
        if self.batch_idx == None:
            return False
        if (
            drop_last_batch
            and self.batch_idx + batch_size > self.buffer_size * self.n_envs
        ):
            return False
        if self.batch_idx > self.buffer_size * self.n_envs:
            return False
        return True

    @partial(jax.jit, static_argnames=["self", "batch_size"])
    def sample_batch(self, key: PRNGKey, batch_size: int, drop_last_batch=True):
        """
        Sample a Batch of data without replacement
        """
        if not self._prepared_for_sampling(batch_size, drop_last_batch):
            self.batch_idx = 0
            if self.full:
                inds = jnp.arange(0, self.buffer_size).repeat(self.n_envs)
            else:
                inds = jnp.arange(0, self.ptr).repeat(self.n_envs)
            env_inds = jnp.tile(jnp.arange(self.n_envs), len(inds) // self.n_envs)
            inds = jnp.vstack([inds, env_inds]).T
            inds = jax.random.shuffle(key=key, x=inds)
            # np.random.shuffle(inds)
            self.batch_inds = inds[:, 0]
            self.batch_env_inds = inds[:, 1]
        batch_ids = self.batch_inds[self.batch_idx : self.batch_idx + batch_size]
        env_ids = self.batch_env_inds[self.batch_idx : self.batch_idx + batch_size]
        self.batch_idx = self.batch_idx + batch_size
        return self._get_batch_by_ids(
            buffers=self.buffers, batch_ids=batch_ids, env_ids=env_ids
        )

    def sample_random_batch(self, rng_key: PRNGKey, batch_size: int):
        """
        Sample a batch of data with replacement
        """
        indx = np.random.randint(self.size(), size=batch_size)
        # np.random.randint
        return self._get_batch_by_ids(
            buffers=self.buffers, batch_ids=indx, env_ids=np.zeros_like(indx)
        )
    # @partial(jax.jit, static_argnames=["self", "batch_size"])
    # def sample_random_batch(self, rng_key: PRNGKey, batch_size: int):
    #     """
    #     Sample a batch of data with replacement
    #     """
    #     # TODO provide faster routine for replay buffers where n_envs = 1?
    #     rng_key, batch_ids_rng_key = jax.random.split(rng_key)
    #     batch_ids = jax.random.randint(
    #         batch_ids_rng_key, shape=(batch_size,), minval=0, maxval=self.buffer_size
    #     )
    #     rng_key, env_ids_rng_key = jax.random.split(rng_key)
    #     env_ids = jax.random.randint(
    #         env_ids_rng_key, shape=(batch_size,), minval=0, maxval=self.n_envs
    #     )
            # bugged, buffers is cached and so no new data is actually retrieved
    #     return self._get_batch_by_ids(
    #         buffers=self.buffers, batch_ids=batch_ids, env_ids=env_ids
    #     )