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

    grad_updates_per_step: Optional[int] = 1
    """
    This defines the max number of gradient updates that are performed after every rollout.
    """

    num_envs: Optional[int] = 1
    """
    Number of parallel envs used. Each training step `step num_envs * steps_per_env` interactions 
    are collected in a rollout before updates are considered
    """

    steps_per_env: Optional[int] = 1
    """
    The number of interaction steps to perform in each parallel environment during a single rollout
    """

    batch_size: Optional[int] = 256
    """
    The size of the batch of rollout data sampled during gradient updates
    """

    normalize_advantage: Optional[bool] = True
    """
    Whether to apply normalization to advantage calculations. 
    """

    discount: Optional[float] = 0.99
    """
    The discount factor
    """

    gae_lambda: Optional[float] = 0.97
    """
    The Generalized Advantage Estimation lambda parameter
    """

    clip_ratio: Optional[float] = 0.2
    """
    The value to clip the log prob ratios (exp(log_p - log_p_old))
    """

    ent_coef: Optional[float] = 0.0
    """
    Entropy coefficient. The entropy loss is `-entropy * ent_coef` and is added to the actor loss. Increase it to "encourage" more exploration
    """

    target_kl: Optional[float] = 0.01
    """
    This `value * 1.5` is used as a threshold over the KL divergence of the updated policy compared to the original one used during a rollout.
    If the KL div is over this threshold, then no more actor updates are made.
    """

    reset_env: Optional[bool] = True
    """
    if False, when collecting interactions we will not reset env directly and carry over env states.
    
    It is useful to set to `False` for solving environments via massive parallelization of interactions (e.g. using Brax).
    This is because to make massive parallelization useful and faster for training, we step through each parallel env much less (and often much less than the max episode length). 
    
    As a result, if we reset after each rollout, our agent will only ever learn to solve the early parts of an environment. 
    Thus by setting reset_env to False, over time the agent's rollouts (assuming each env may reset at various times) will 
    be a more diverse collection of behaviors at different time points in the environment.

    The env states are carried over directly by keeping track of states if they are jax based envs. Otherwise they are carried over by simply not resetting the environment between
    training steps
    """

    eval_freq: Optional[int] = 20_000
    """
    Every eval_freq interactions an evaluation is performed
    """

    eval_steps: Optional[int] = 1_000
    """
    Number of evaluation steps taken for each eval environment
    """

    num_eval_envs: Optional[int] = 4
    """
    Number of evaluation envs to use
    """

    log_freq: Optional[int] = 1
    """
    Every log_freq interactions metrics are logged
    """

    save_freq: Optional[int] = 20_000
    """
    Every save_freq interactions the current training state is saved.
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
