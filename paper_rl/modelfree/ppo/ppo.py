import math
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
from paper_rl.common.rollout import Rollout
from paper_rl.common.utils import to_torch
from paper_rl.logger.logger import Logger
from paper_rl.modelfree.ppo.buffer import PPOBuffer


class PPO:
    def __init__(
        self,
        ac: ActorCritic,
        env: gym.Env,
        num_envs: int,
        observation_space,
        action_space,
        steps_per_epoch: int = 10000,
        train_iters: int = 80,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        # max_grad_norm: float = 0.5, # TODO
        # use_sde: bool = False,
        # sde_sample_freq: int = -1,
        target_kl: Optional[float] = 0.01,
        logger: Logger = None,
        # create_eval_env: bool = False,
        # verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cpu",
        # _init_setup_model: bool = True
    ) -> None:
        # Random seed
        if seed is None:
            seed = 0
        # seed += 10000 * proc_id()
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        self.n_envs = num_envs
        self.env = env  # should be vectorized
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space

        # hparams
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # self.pi_optimizer = optim.Adam(ac.pi.parameters(), lr=pi_lr)
        # self.vf_optimizer = optim.Adam(ac.v.parameters(), lr=vf_lr)

        # exp params
        self.train_iters = train_iters
        self.steps_per_epoch = steps_per_epoch

        self.logger = logger
        self.buffer = PPOBuffer(
            buffer_size=self.steps_per_epoch,
            observation_space=observation_space,
            action_space=action_space,
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

    def train(
        self,
        train_callback=None,
        rollout_callback=None,
        obs_to_tensor=None,
        max_ep_len=None,
        start_epoch: int = 0,
        n_epochs: int = 10,
        critic_warmup_epochs: int = 0, # number of epochs to collect rollouts and update critic only, freezing policy
        pi_optimizer: torch.optim.Optimizer = None,
        vf_optimizer: torch.optim.Optimizer = None,
        optim=None,
        batch_size=1000,
        compute_delta_loss=False,
        accumulate_grads=False,
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
        n_envs = self.n_envs
        device = self.device
        rollout = Rollout()
        def policy(o):
            if obs_to_tensor is None:
                o = torch.as_tensor(o, dtype=torch.float32)
            return ac.step(o)

        def update(update_pi=True):
            data = buf.get()
            update_res = ppo_update(
                data=data,
                ac=ac,
                pi_optimizer=pi_optimizer,
                vf_optimizer=vf_optimizer,
                clip_ratio=clip_ratio,
                train_iters=train_iters,
                batch_size=batch_size,
                target_kl=target_kl,
                logger=logger,
                compute_old=compute_delta_loss,
                device=device,
                accumulate_grads=accumulate_grads,
                update_pi=update_pi
            )
            pi_info, loss_pi, loss_v, pi_l_old, v_l_old, update_step = (
                update_res["pi_info"],
                update_res["loss_pi"],
                update_res["loss_v"],
                update_res["pi_l_old"],
                update_res["v_l_old"],
                update_res["update_step"],
            )
            logger.store(tag="train", StopIter=update_step, append=False)
            
            if loss_v is not None:
                logger.store(
                    tag="train",
                    LossV=loss_v.cpu().item(),
                )
            if pi_info is not None:
                kl, ent, cf = pi_info["kl"], pi_info["ent"], pi_info["cf"]
                logger.store(
                    tag="train",
                    LossPi=loss_pi.cpu().item(),
                    KL=kl,
                    Entropy=ent,
                    ClipFrac=cf,
                )
            if compute_delta_loss:
                logger.store(
                    tag="train",
                    DeltaLossPi=(loss_pi.cpu().item() - pi_l_old),
                    DeltaLossV=(loss_v.cpu().item() - v_l_old),
                )

        # to tweak, just copy the code below
        for epoch in range(start_epoch, start_epoch + n_epochs):
            rollout_start_time = time.time_ns()
            ac.pi.eval()
            rollout.collect(policy=policy, env=env, n_envs=n_envs, buf=buf, steps=self.steps_per_epoch, rollout_callback=rollout_callback, max_ep_len=max_ep_len, logger=logger)
            ac.pi.train()
            rollout_end_time = time.time_ns()
            rollout_delta_time = (rollout_end_time - rollout_start_time) * 1e-9
            logger.store("train", RolloutTime=rollout_delta_time, critic_warmup_epochs=critic_warmup_epochs, append=False)
            update_start_time = time.time_ns()
            update_pi = epoch >= critic_warmup_epochs
            update(update_pi = update_pi)
            update_end_time = time.time_ns()
            logger.store("train", UpdateTime=(update_end_time - update_start_time) * 1e-9, append=False)
            logger.store("train", Epoch=epoch, append=False)
            logger.store("train", TotalEnvInteractions=self.steps_per_epoch * self.n_envs * (epoch + 1), append=False)
            stats = logger.log(step=epoch)
            logger.reset()
            if train_callback is not None:
                train_callback(epoch=epoch, stats=stats)


def ppo_update(
    data,
    ac: ActorCritic,
    pi_optimizer,
    vf_optimizer,
    clip_ratio,
    train_iters,
    batch_size,
    target_kl,
    logger=None,
    compute_old=False,
    device=torch.device("cpu"),
    accumulate_grads=False,
    update_pi=True,
):
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"].to(device), data["logp"].to(device)
        # if isinstance(self.env.action_space, spaces.Discrete):
        # Convert discrete action from float to long
        # print(act.shape, act[:2])
        # act = act.long().flatten()
        # print(act.shape, act[:2])

        # Policy loss)
        ac.pi.eval()
        pi, logp = ac.pi(obs, act)
        ac.pi.train()
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Entropy loss for some basic extra exploration
        entropy = pi.entropy()
        with torch.no_grad():
            # Useful extra info
            logr = logp - logp_old
            approx_kl = (torch.exp(logr) - 1 - logr).mean().cpu().item()
            
            # approx_kl = (logp_old - logp).mean().item()
            clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy.mean().item(), cf=clipfrac)
        return loss_pi, logp, entropy, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"].to(device)
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_l_old, v_l_old, entropy_old = None, None, None
    if compute_old:
        pi_l_old, logp_old, entropy_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

    # Train policy with multiple steps of gradient descent
    update_step = 0
    early_stop_update = False
    loss_pi = None
    pi_info = None
    for _ in range(train_iters):
        if early_stop_update:
            break
        N = len(data["obs"])
        # print(data["obs"].shape, data["adv"].shape)

        steps_per_train_iter = int(math.ceil(N / batch_size))
        if accumulate_grads:
            if update_pi: pi_optimizer.zero_grad()
            vf_optimizer.zero_grad()
            average_kl = 0
        for batch_idx in range(steps_per_train_iter):
            batch_data = dict()
            for k, v in data.items():
                batch_data[k] = v[max(0, (batch_idx) * batch_size) : (batch_idx + 1) * batch_size]
            if update_pi:
                loss_pi, logp, entropy, pi_info = compute_loss_pi(batch_data)
                kl = pi_info["kl"]
                if accumulate_grads:
                    average_kl += kl
                if not accumulate_grads and target_kl is not None:
                    if kl > 1.5 * target_kl:
                        logger.print("Early stopping at step %d due to reaching max kl." % update_step)
                        early_stop_update = True
                        break
            loss_v = compute_loss_v(batch_data)
            # TODO - entropy loss
            # if entropy is None:
            #     # Approximate entropy when no analytical form
            #     entropy_loss = -torch.mean(-logp)
            # else:
            #     entropy_loss = -torch.mean(entropy)
            # torch.nn.utils.clip_grad.clip_grad_norm_() # TODO
            if not accumulate_grads:
                if update_pi:
                    pi_optimizer.zero_grad()
                vf_optimizer.zero_grad()
                if update_pi:
                    loss_pi.backward()
                    pi_optimizer.step()
                loss_v.backward()
                vf_optimizer.step()
                update_step += 1
            if accumulate_grads:
                # scale loss down
                if update_pi: loss_pi /= steps_per_train_iter
                loss_v /= steps_per_train_iter
                if update_pi: loss_pi.backward()
                loss_v.backward()
        
        if accumulate_grads and target_kl is not None:
            average_kl /= steps_per_train_iter
            if average_kl > 1.5 * target_kl:
                logger.print("Early stopping at step %d due to reaching max kl." % update_step)
                early_stop_update = True
                break
        if accumulate_grads:
            if update_pi: pi_optimizer.step()
            vf_optimizer.step()
            update_step += 1
        if early_stop_update:
            break
    return dict(
        pi_info=pi_info,
        update_step=update_step,
        loss_v=loss_v,
        loss_pi=loss_pi,
        pi_l_old=pi_l_old,
        v_l_old=v_l_old,
        entropy_old=entropy_old,
    )
