import warnings
from typing import Any, Dict, Optional, Type, Union

import gym
import numpy as np
import torch
import torch as th
from gym import spaces
from torch import optim
from torch.nn import functional as F

from paper_rl.architecture.ac.core import ActorCritic, count_vars
from paper_rl.common.mpi.mpi_pytorch import (mpi_avg_grads,
                                             setup_pytorch_for_mpi,
                                             sync_params)
from paper_rl.common.mpi.mpi_tools import proc_id
from paper_rl.logger.logger import Logger
from paper_rl.modelfree.ppo.buffer import PPOBuffer


class PPO:
    def __init__(
        self,
        ac: ActorCritic,
        env: gym.Env,
        num_envs: int,
        pi_lr: float = 3e-4,
        vf_lr: float = 3e-4,
        steps_per_epoch: int = 10000,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        # max_grad_norm: float = 0.5,
        # use_sde: bool = False,
        # sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        logger: Logger = None,
        create_eval_env: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cpu",
        # _init_setup_model: bool = True
    ) -> None:
        # Random seed
        if seed is None:
            seed = 0
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.n_envs = num_envs
        self.env = env  # should be vectorized
        self.device = device

        # hparams
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.pi_optimizer = optim.Adam(ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(ac.v.parameters(), lr=vf_lr)

        # exp params
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.start_epoch = 0

        self.buffer = PPOBuffer(
            buffer_size=self.steps_per_epoch * num_envs,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            n_envs=self.n_envs,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )
        self.ac = ac.to(self.device)
        var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
        self.logger.print(
            "\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts,
            color="green",
            bold=True,
        )

    def to_state_dict(self):
        pass

    def train(self, train_callback=None):
        for epoch in range(self.start_epoch, self.start_epoch + self.n_epochs):

            pass
