"""
Configurations and utility classes
"""

import dataclasses
from typing import Optional

import chex
from flax import struct


@dataclasses.dataclass
class SACConfig:
    """
    Configuration dataclass for SAC
    """
    max_episode_length: Optional[int] = -1 # IF this value is not set, we expect during training the steps_per_epoch is >= max episode length of the environment
    normalize_advantage: Optional[bool] = True
    tau: Optional[float] = 0.005
    gamma: Optional[float] = 0.99
    alpha: Optional[float] = 0.2


@struct.dataclass
class TimeStep:
    log_p: chex.Array = None
    action: chex.Array = None
    env_obs: chex.Array = None
    adv: chex.Array = None
    reward: chex.Array = None
    orig_ret: chex.Array = None
    ep_ret: chex.Array = None
    value: chex.Array = None
    done: chex.Array = None
    ep_len: chex.Array = None
    info: chex.Array = None
