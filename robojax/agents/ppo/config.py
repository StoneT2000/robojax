"""
Configurations and utility classes
"""

import dataclasses
from typing import Optional

import chex
from flax import struct


@dataclasses.dataclass
class PPOConfig:
    """
    Configuration dataclass for PPO
    """
    update_iters: int
    """
    This defines the max number of gradient updates that are performed after every rollout.
    """

    steps_per_epoch: int
    """
    The number of interaction steps to perform in each parallel environment during a single rollout
    """

    num_envs: int
    """
    The number of parallel environments
    """

    batch_size: int
    """
    The size of the batch of rollout data sampled during gradient updates
    """

    normalize_advantage: Optional[bool] = True
    gamma: Optional[float] = 0.99
    gae_lambda: Optional[float] = 0.97
    clip_ratio: Optional[float] = 0.2
    ent_coef: Optional[float] = 0.0
    pi_coef: Optional[float] = 1.0
    vf_coef: Optional[float] = 1.0
    dapg_lambda: Optional[float] = 0.1
    dapg_damping: Optional[float] = 0.99
    target_kl: Optional[float] = 0.01
    reset_env: Optional[bool] = True
    """
    if False, when collecting interactions we will not reset env directly and carry over env states.
    
    It is useful to set to False for solving environments via massive parallelization of interactions (e.g. using Brax).
    This is because to make massive parallelization useful and faster for training, we step through each parallel env much less (and often much less than the max episode length). 
    As a result, if we reset after each rollout, our agent will only ever learn to solve the early parts of an environment. 
    Thus by setting reset_env to False, over time the agent's rollouts (assuming each env may reset at various times) will 
    be a more diverse collection of behaviors at different time points in the environment.
    """


    eval_freq: Optional[int] = 10
    """
    Every eval_freq training steps (composed of rollout and update) an evaluation is performed
    """
    eval_steps: Optional[int] = 1000
    """
    Number of evaluation steps taken for each eval environment
    """
    num_eval_envs: Optional[int] = 4
    """
    Number of evaluation envs to use
    """

    log_freq: Optional[int] = 1
    """
    Every log_freq training steps metrics (e.g. TODO) are logged
    """
    save_freq: Optional[int] = 10
    """
    Every save_freq training steps the current training state is saved.
    """

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
