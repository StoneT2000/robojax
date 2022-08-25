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

    num_seed_steps: int
    num_train_steps: int
    replay_buffer_capacity: int

    batch_size: int

    tau: Optional[float] = 0.005
    discount: Optional[float] = 0.99
    backup_entropy: Optional[bool] = True
    target_entropy: Optional[float] = None  # defaults to - act_dims / 2
    learnable_temp: Optional[bool] = True
    initial_temperature: Optional[float] = 1.0
    
    actor_update_freq: Optional[int] = 1
    target_update_freq: Optional[int] = 1

    eval_freq: Optional[int] = 5000
    eval_steps: Optional[int] = 1000
    num_eval_envs: Optional[int] = 4


    log_freq: Optional[int] = 1000

    max_episode_length: Optional[int] = -1


@struct.dataclass
class TimeStep:
    action: chex.Array = None
    env_obs: chex.Array = None
    next_env_obs: chex.Array = None
    reward: chex.Array = None
    mask: chex.Array = None
