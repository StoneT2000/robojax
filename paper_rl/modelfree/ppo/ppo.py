import math
from re import I
import time
import warnings
from typing import Any, Dict, Optional, Type, Union

import gym
import numpy as np
import torch
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
        train_iters: int = 80,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        # max_grad_norm: float = 0.5, # TODO
        # use_sde: bool = False,
        # sde_sample_freq: int = -1,
        target_kl: Optional[float] = 0.01,
        logger: Logger = None,
        create_eval_env: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cpu",
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
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.pi_optimizer = optim.Adam(ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(ac.v.parameters(), lr=vf_lr)

        # exp params
        self.train_iters = train_iters
        self.steps_per_epoch = steps_per_epoch
        

        self.logger = logger
        self.buffer = PPOBuffer(
            size=self.steps_per_epoch,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            # n_envs=self.n_envs,
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

    def train(
        self, 
        train_callback=None, 
        max_ep_len=None,
        start_epoch: int = 0,
        n_epochs: int = 10,
        pi_optimizer: torch.optim.Optimizer = None,
        vf_optimizer: torch.optim.Optimizer = None,
        optim = None,
        batch_size=1000,
        compute_delta_loss=False,
    ):
        if max_ep_len is None:
            # TODO: infer this
            raise ValueError("max_ep_len is missing") 
        ac = self.ac
        env = self.env
        buf = self.buffer
        logger = self.logger
        clip_ratio = self.clip_ratio
        train_iters = self.train_iters
        target_kl = self.target_kl

        # Set up function for computing PPO policy loss
        # def compute_loss_pi(data):
        #     obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        #     # if isinstance(self.env.action_space, spaces.Discrete):
        #         # Convert discrete action from float to long
        #         # print(act.shape, act[:2])
        #         # act = act.long().flatten()
        #         # print(act.shape, act[:2])

        #     # Policy loss
        #     pi, logp = ac.pi(obs, act)
        #     ratio = torch.exp(logp - logp_old)
        #     clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        #     loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        #     # Entropy loss for some basic extra exploration
        #     entropy = pi.entropy()

        #     # Useful extra info
        #     approx_kl = (logp_old - logp).mean().item()
            
        #     clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        #     clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        #     pi_info = dict(kl=approx_kl, ent=entropy.mean().item(), cf=clipfrac)

        #     return loss_pi, logp, entropy, pi_info

        # # Set up function for computing value loss
        # def compute_loss_v(data):
        #     obs, ret = data['obs'], data['ret']
        #     return ((ac.v(obs) - ret)**2).mean()

        def compute_loss_pi(data):
            obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

            # Policy loss
            pi, logp = ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # Useful extra info
            # TODO swap this with http://joschu.net/blog/kl-approx.html ?
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            return loss_pi, pi_info

        # Set up function for computing value loss
        def compute_loss_v(data):
            obs, ret = data["obs"], data["ret"]
            return ((ac.v(obs) - ret) ** 2).mean()
        def update():
            data = buf.get()
            pi_l_old, v_l_old=None, None
            if compute_delta_loss:
                pi_l_old,logp_old, entropy_old, pi_info_old = compute_loss_pi(data)
                pi_l_old = pi_l_old.item()
                v_l_old = compute_loss_v(data).item()

            # Train policy with multiple steps of gradient descent
            update_step = 0
            early_stop_update = False
            for _ in range(train_iters):
                if early_stop_update: break
                N = len(data["obs"])
                # print(data["obs"].shape, data["adv"].shape)
                steps_per_train_iter = int(math.ceil(N / batch_size))
                for batch_idx in range(steps_per_train_iter):
                    batch_data = dict()
                    for k, v in data.items():
                        batch_data[k] = v[max(0, (batch_idx) * batch_size): (batch_idx+1) * batch_size]
                    loss_pi, pi_info = compute_loss_pi(batch_data)
                    loss_v = compute_loss_v(batch_data)
                    kl = pi_info['kl']
                    if target_kl is not None:
                        if kl > 1.5 * target_kl:
                            logger.print('Early stopping at step %d due to reaching max kl.'%update_step)
                            early_stop_update = True
                            break

                    # if entropy is None:
                    #     # Approximate entropy when no analytical form
                    #     entropy_loss = -torch.mean(-logp)
                    # else:
                    #     entropy_loss = -torch.mean(entropy)
                    # # vf_optimizer.zero_grad()
                    # # pi_optimizer.zero_grad()
                    # # loss = loss_pi + self.ent_coef * entropy_loss
                    # # loss_v.backward()
                    # # loss.backward()
                    # # torch.nn.utils.clip_grad.clip_grad_norm_() # TODO
                    # # pi_optimizer.step()
                    # # vf_optimizer.step()
                    optim.zero_grad()
                    loss = loss_pi + self.vf_coef * loss_v
                    # loss = loss_pi + self.ent_coef * entropy_loss + self.vf_coef * loss_v
                    loss.backward()
                    optim.step()
                    update_step += 1

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
            if compute_delta_loss:
                logger.store(
                    tag="train",
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    DeltaLossV=(loss_v.item() - v_l_old),
                )


        observations, ep_returns, ep_lengths = env.reset(), 0, 0
        # observations, ep_returns, ep_lengths = env.reset(), np.zeros(self.n_envs), np.zeros(self.n_envs)
        # to tweak, just copy the code below
        for epoch in range(start_epoch, start_epoch + n_epochs):
            rollout_start_time = time.time_ns()
            for t in range(self.steps_per_epoch):
                a, v, logp = ac.step(torch.as_tensor(observations, dtype=torch.float32))
                next_os, rewards, dones, infos = env.step(a)
                # for idx, d in enumerate(dones):
                #     ep_returns[idx] += returns[idx]
                ep_returns += rewards
                ep_lengths += 1
                buf.store(observations, a, rewards, v, logp)
                logger.store(tag="train", VVals=v)

                observations = next_os

                timeouts = ep_lengths == max_ep_len
                terminals = dones | timeouts # terminated means done or reached max ep length
                epoch_ended = t == self.steps_per_epoch - 1

                for idx, terminal in enumerate(terminals):
                    if terminal or epoch_ended:
                        if "terminal_observation" in infos[idx]:
                            o = infos[idx]["terminal_observation"]
                        else:
                            o = observations[idx]
                        ep_ret = ep_returns[idx]
                        ep_len = ep_lengths[idx]
                        timeout = timeouts[idx]
                        if epoch_ended and not terminal:
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_lengths[idx], flush=True)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        if timeout or epoch_ended:
                            _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                        else:
                            v = 0
                        buf.finish_path(idx, v)
                        if terminal:
                            # only save EpRet / EpLen if trajectory finished
                            logger.store("train", EpRet=ep_ret, EpLen=ep_len)
                        ep_returns[idx] = 0
                        ep_lengths[idx] = 0
                        # observations, ep_ret, ep_len = env.reset(), np.zeros(self.n_envs), np.zeros(self.n_envs)
            rollout_end_time = time.time_ns()
            rollout_delta_time = (rollout_end_time - rollout_start_time) * 1e-9
            logger.store("train", RolloutTime=rollout_delta_time, append=False)
            update_start_time = time.time_ns()
            update()
            update_end_time = time.time_ns()
            logger.store("train", UpdateTime=(update_end_time - update_start_time) * 1e-9, append=False)
            logger.store("train", Epoch=(epoch), append=False)
            stats = logger.log(step=epoch)
            filtered_stats = {"epoch": epoch}
            for k in stats.keys():
                if "Ret" in k or "Len" in k:
                    filtered_stats[k] = stats[k]
            logger.pretty_print_table(filtered_stats)

