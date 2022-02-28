from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
from gym import spaces

from paper_rl.common.buffer import BaseBuffer
from paper_rl.common.mpi.mpi_tools import mpi_statistics_scalar
from paper_rl.common.stats import discount_cumsum

class PPOBuffer(BaseBuffer):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        gamma=0.99,
        lam=0.95,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
        )
        self.obs_is_dict = False
        if isinstance(self.obs_shape, dict):
            # raise NotImplementedError("Can't handle dict observations yet!")
            self.obs_is_dict = True
            self.obs_buf = {}
            for k in self.observation_space:
                self.obs_buf[k] = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape[k], dtype=self.observation_space[k].dtype)
        else:
            self.obs_buf = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32,
                # device=self.device
            )
        self.act_buf = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32,
            # device=self.device
        )
        self.adv_buf = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32,
        # device=self.device
        )
        self.rew_buf = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32, 
        # device=self.device
        )
        self.ret_buf = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32, 
        # device=self.device
        )
        self.val_buf = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32, 
        # device=self.device
        )
        self.logp_buf = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32, 
        # device=self.device
        )
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, [0]*n_envs, self.buffer_size
        self.next_batch_idx = 0

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        if self.obs_is_dict:
            for k in obs.keys():
                obs[k] = obs[k].reshape(self.obs_buf[k][self.ptr].shape)
                self.obs_buf[k][self.ptr] = np.array(obs[k]).copy()
        else:
            self.obs_buf[self.ptr] = np.array(obs).copy()
        if isinstance(self.action_space, spaces.Discrete):
            act = act.reshape((self.n_envs, self.action_dim))
        self.act_buf[self.ptr] = np.array(act).copy()
        self.rew_buf[self.ptr] = np.array(rew).copy()
        self.val_buf[self.ptr] = np.array(val).copy()
        self.logp_buf[self.ptr] = np.array(logp).copy()
        self.ptr += 1

    def finish_path(self, env_id, last_val=0):
        # TODO: remove env_id and do it in batch
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx[env_id], self.ptr)
        rews = np.append(self.rew_buf[path_slice, env_id], last_val)
        vals = np.append(self.val_buf[path_slice, env_id], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # print(deltas.shape, self.adv_buf.shape, self.adv_buf[path_slice,env_id].shape)
        self.adv_buf[path_slice, env_id] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice, env_id] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx[env_id] = self.ptr
    # def sample(self, batch_size):
    #     self.next_batch_idx: self.next_batch_idx + batch_size
    # def reset(self):
    #     self.ptr, self.path_start_idx = 0, 0
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        N = self.buffer_size * self.n_envs
        self.ptr, self.path_start_idx = 0, [0] * self.n_envs
        # the next line implement the advantage normalization trick
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std())
        data = dict(
            # obs=
            ret=self.ret_buf.reshape(-1),
            adv=self.adv_buf.reshape(-1),
            logp=self.logp_buf.reshape(-1),
        )
        if self.obs_is_dict:
            # flattened = [item for sublist in self.obs_buf for item in sublist]
            # print("OBSBUF", len(self.obs_buf), len(flattened))
            # data["obs"] = flattened
            data["obs"] = {}
            for k in self.obs_shape:
                data["obs"][k] = self.obs_buf[k].reshape((-1, ) + self.obs_shape[k])
                self.obs_buf[k] = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape[k], dtype=self.observation_space[k].dtype)
        else:
            data["obs"] = self.obs_buf.reshape((-1, ) + self.obs_shape)
        if isinstance(self.action_space, spaces.Discrete):
            data["act"] = self.act_buf.reshape(-1)
        else:
            data["act"] = self.act_buf.reshape((-1, self.action_dim))
        tensored_data = {k: torch.as_tensor(data[k], dtype=torch.float32) for k in ["ret", "adv", "logp", "act"]}
        if self.obs_is_dict:
            # tensored_data["obs"] = torch.as_tensor(data["obs"], dtype=torch.float32)
            for k in data["obs"]:
                data["obs"][k] = torch.as_tensor(data["obs"][k])
            tensored_data["obs"] = data["obs"]
        else:
            tensored_data["obs"] = torch.as_tensor(data["obs"], dtype=torch.float32)
        return tensored_data
