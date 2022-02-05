import math
import time
import warnings
from typing import Any, Dict, Optional, Type, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch import optim
from torch.nn import functional as F

from paper_rl.architecture.ac.core import ActorCritic, count_vars
from paper_rl.common.rollout import Rollout
from paper_rl.logger.logger import Logger
from paper_rl.modelfree.ppo.buffer import PPOBuffer
from paper_rl.modelfree.ppo.ppo import ppo_update

class GAIL():
    """
    TODO - make compatible with other policies
    """
    def __init__(
        self,
        ac: ActorCritic,
        discriminator: nn.Module,
        env: gym.Env,
        num_envs: int,
        observation_space,
        action_space,
        pi_lr: float = 3e-4,
        vf_lr: float = 3e-4,
        steps_per_epoch: int = 10000,
        train_iters: int = 80,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        # max_grad_norm: float = 0.5, # TODO
        # use_sde: bool = False,
        # sde_sample_freq: int = -1,
        target_kl: Optional[float] = 0.01,
        logger: Logger = None,
        create_eval_env: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        self.n_envs = num_envs
        self.env = env  # should be vectorized
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space

        # hparams
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # exp params
        self.train_iters = train_iters
        self.steps_per_epoch = steps_per_epoch
        self.buffer = PPOBuffer(
            buffer_size=self.steps_per_epoch,
            observation_space=observation_space,
            action_space=action_space,
            n_envs=self.n_envs,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )
        self.logger = logger
        self.ac = ac.to(self.device)
        var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
        self.logger.print(
            "\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts,
            color="green",
            bold=True,
        )
    def train(
        self, 
        train_callback=None,
        rollout_callback=None,
        max_ep_len=None,
        start_epoch: int = 0,
        n_epochs: int = 10,
        pi_optimizer: torch.optim.Optimizer = None,
        vf_optimizer: torch.optim.Optimizer = None,
        discrim_optimizer: torch.optim.Optimizer = None,
        batch_size=32,
        ppo_batch_size=128,
        compute_delta_loss=False,
        expert_reward=None,
        sample_expert_trajectories=None,
    ):
        assert expert_reward is not None
        assert sample_expert_trajectories is not None
        ac = self.ac
        env = self.env
        buf = self.buffer
        logger = self.logger
        clip_ratio = self.clip_ratio
        target_kl = self.target_kl
        discriminator = self.discriminator
        train_iters = self.train_iters
        n_envs = self.n_envs

        discriminator_criterion = nn.BCELoss()
        rollout = Rollout()
        def policy(o):
            o = torch.as_tensor(o, dtype=torch.float32)
            return ac.step(o)
        for epoch in range(start_epoch, n_epochs + start_epoch):
            # sample trajectories T_i
            rollout_start_time = time.time_ns()
            rollout.collect(policy=policy, env=env, n_envs=n_envs, buf=buf, steps=self.steps_per_epoch, rollout_callback=rollout_callback, max_ep_len=max_ep_len, logger=logger,custom_reward=expert_reward)
            rollout_end_time = time.time_ns()
            rollout_delta_time = (rollout_end_time - rollout_start_time) * 1e-9
            logger.store("train", RolloutTime=rollout_delta_time, append=False)

            data = buf.get()
            rollout_obs = data["obs"]
            rollout_act = data["act"]

            # update discriminator
            disc_update_start_time = time.time_ns()
            for _ in range(1):
                # sample some expert trajectories
                expert_trajectories = sample_expert_trajectories(batch_size)
                # expert_trajectories["observations"] - (B, obs_space)
                # expert_trajectories["actions"] - (B, act_dim)
                g_o = discriminator(rollout_obs, rollout_act)
                e_o = discriminator(expert_trajectories["observations"], expert_trajectories["actions"])
                discrim_optimizer.zero_grad()
                discrim_loss = discriminator_criterion(g_o, torch.ones((rollout_obs.shape[0], 1), device=self.device)) + \
                    discriminator_criterion(e_o, torch.zeros((expert_trajectories["observations"].shape[0], 1), device=self.device))
                discrim_loss.backward()
                discrim_optimizer.step()
            disc_update_end_time = time.time_ns()
            logger.store("train", DiscriminatorUpdateTime=(disc_update_end_time - disc_update_start_time) * 1e-9, append=False)
            
            # update via ppo
            ppo_update_start_time = time.time_ns()
            update_res = ppo_update(
                data,
                ac=ac,
                pi_optimizer=pi_optimizer,
                vf_optimizer=vf_optimizer,
                clip_ratio=clip_ratio,
                train_iters=train_iters,
                batch_size=ppo_batch_size,
                target_kl=target_kl,
                logger=logger,
                compute_old=False,
            )
            ppo_update_end_time = time.time_ns()
            pi_info, loss_pi, loss_v, pi_l_old, v_l_old, update_step = (
                update_res["pi_info"],
                update_res["loss_pi"],
                update_res["loss_v"],
                update_res["pi_l_old"],
                update_res["v_l_old"],
                update_res["update_step"],
            )
            logger.store(tag="train", StopIter=update_step, append=False)
            kl, ent, cf = pi_info["kl"], pi_info["ent"], pi_info["cf"]
            logger.store(
                tag="train",
                LossPi=loss_pi.item(),
                LossV=loss_v.item(),
                KL=kl,
                Entropy=ent,
                ClipFrac=cf,
            )

            logger.store("train", PPOUpdateTime=(ppo_update_end_time - ppo_update_start_time) * 1e-9, append=False)
            logger.store("train", Epoch=epoch, append=False)
            logger.store("train", TotalEnvInteractions=self.steps_per_epoch * self.n_envs * (epoch + 1), append=False)
            stats = logger.log(step=epoch)

            if train_callback is not None: 
                train_callback(epoch=epoch, stats=stats)