"""
Configurations and utility classes
"""

import dataclasses
from optparse import Option
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

    num_envs: Optional[int] = 1

    # Various SAC hyperparameters
    steps_per_env: Optional[int] = 1
    """
    Usually SAC steps once through every environment before performing a gradient update. 
    You can change steps_per_env to increase the number of steps performed for each training step
    """
    grad_updates_per_step: Optional[int] = 1
    """
    Number of gradient updates for each training step.
    """
    
    tau: Optional[float] = 0.005
    discount: Optional[float] = 0.99
    backup_entropy: Optional[bool] = True
    target_entropy: Optional[float] = None  # defaults to - act_dims / 2
    learnable_temp: Optional[bool] = True
    initial_temperature: Optional[float] = 1.0
    actor_update_freq: Optional[int] = 1
    target_update_freq: Optional[int] = 1

    eval_freq: Optional[int] = 5000
    """
    Every eval_freq training steps (composed of env step and update) an evaluation is performed
    """
    eval_steps: Optional[int] = 1000
    """
    Number of evaluation steps taken for each eval environment
    """
    num_eval_envs: Optional[int] = 4
    """
    Number of evaluation envs to use
    """

    log_freq: Optional[int] = 1000
    """
    Every log_freq training steps metrics (e.g. critic loss) are logged
    """
    save_freq: Optional[int] = 100_000
    """
    Every save_freq training steps the current training state is saved.
    """

    max_episode_length: Optional[int] = None


@struct.dataclass
class TimeStep:
    action: chex.Array = None
    env_obs: chex.Array = None
    next_env_obs: chex.Array = None
    reward: chex.Array = None
    mask: chex.Array = None
