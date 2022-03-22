import time

import gym
import numpy as np
import torch

from paper_rl.common.buffer import BaseBuffer

from tqdm import tqdm
class Rollout:
    def __init__(self) -> None:
        pass
    def collect_trajectories(
        self,
        policy,
        env: gym.Env,
        n_trajectories,
        n_envs,
        rollout_callback=None,
        custom_reward=None,
        logger=None,
        render=False,
        pbar=False,
        even_num_traj_per_env=False # collects even number of trajectories per env
    ):
        traj_count = 0
        if even_num_traj_per_env:
            assert n_trajectories % n_envs == 0
            max_trajectories_per_env = n_trajectories / n_envs
            trajectories_per_env = []
        trajectories = []
        past_obs = []
        past_rews = []
        past_acts = []
        for _ in range(n_envs):
            past_obs.append([])
            past_rews.append([])
            past_acts.append([])
            if even_num_traj_per_env:
                trajectories_per_env.append([])
        observations = env.reset()
        is_dict = isinstance(observations, dict)
        step = 0
        if pbar:
            pbar = tqdm(total=n_trajectories)
        while True:
            with torch.no_grad():
                acts = policy(observations)

            if is_dict:
                for idx in range(n_envs):
                    o = {}
                    for k in observations.keys():
                        o[k] = observations[k][idx]
                    past_obs[idx].append(o)
            else:
                for idx, o in enumerate(observations):
                    past_obs[idx].append(o)
            for idx, a in enumerate(acts):
                past_acts[idx].append(a)
            if render: env.render()
            next_os, rewards, dones, infos = env.step(acts)
            if custom_reward is not None: rewards = custom_reward(rewards, observations, acts)
            for idx, r in enumerate(rewards):
                past_rews[idx].append(r)
            if rollout_callback is not None:
                rollout_callback(
                    observations=observations,
                    next_observations=next_os,
                    actions=acts,
                    rewards=rewards,
                    infos=infos,
                    dones=dones,
                )

            observations = next_os
            # eval_env.render()
            for idx, d in enumerate(dones):
                if d:
                    if not even_num_traj_per_env or len(trajectories_per_env[idx]) < max_trajectories_per_env:
                        termminal_obs = infos[idx]["terminal_observation"]
                        past_obs[idx].append(termminal_obs)
                        t_obs = past_obs[idx] if is_dict else np.vstack(past_obs[idx])
                        t_act = np.vstack(past_acts[idx])
                        t_rew = np.vstack(past_rews[idx])
                        traj = {
                            "observations": t_obs,
                            "rewards": t_rew,
                            "actions": t_act
                        }
                        traj_count += 1
                        if not even_num_traj_per_env:
                            trajectories.append(traj)
                        else:
                            trajectories_per_env[idx].append(traj)
                    past_obs[idx] = []
                    past_acts[idx] = []
                    past_rews[idx] = []
                    if pbar: pbar.update()
                    if traj_count == n_trajectories:
                        if even_num_traj_per_env: return trajectories_per_env
                        else: return trajectories
            step += 1
    def collect(
        self,
        policy,
        env: gym.Env,
        buf: BaseBuffer,
        steps,
        n_envs,
        rollout_callback=None,
        max_ep_len=1000,
        custom_reward=None,
        logger=None,
    ):
        """
        collects for a buffer
        """
        # policy should return a, v, logp
        observations, ep_returns, ep_lengths = env.reset(), np.zeros(n_envs), np.zeros(n_envs, dtype=int)
        is_dict = isinstance(observations, dict)
        rollout_start_time = time.time_ns()
        for t in range(steps):
            a, v, logp = policy(observations)
            next_os, rewards, dones, infos = env.step(a)
            if custom_reward is not None: rewards = custom_reward(rewards, observations, a)
            # for idx, d in enumerate(dones):
            #     ep_returns[idx] += returns[idx]
            ep_returns += rewards
            ep_lengths += 1
            buf.store(obs_buf=observations, act_buf=a, rew_buf=rewards, val_buf=v, logp_buf=logp)
            timeouts = ep_lengths == max_ep_len
            terminals = dones | timeouts  # terminated means done or reached max ep length
            epoch_ended = t == steps - 1
            if rollout_callback is not None:
                rollout_callback(
                    observations=observations,
                    next_observations=next_os,
                    actions=a,
                    rewards=rewards,
                    infos=infos,
                    dones=dones,
                    timeouts=timeouts,
                )
            if logger is not None: logger.store(tag="train", VVals=v)

            observations = next_os
            for idx, terminal in enumerate(terminals):
                if terminal or epoch_ended:
                    if "terminal_observation" in infos[idx]:
                        # batch the input
                        o = infos[idx]["terminal_observation"]
                        if is_dict:
                            for k in o:
                                o[k] = np.expand_dims(o[k], axis=0)
                    else:
                        if is_dict:
                            o = {}
                            for k in observations:
                                o[k] = observations[k][idx:idx+1]
                        else:
                            o = observations[idx]
                    ep_ret = ep_returns[idx]
                    ep_len = ep_lengths[idx]
                    timeout = timeouts[idx]
                    if epoch_ended and not terminal:
                        print("Warning: trajectory cut off by epoch at %d steps." % ep_lengths[idx], flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = policy(o)
                    else:
                        v = 0
                    buf.finish_path(idx, v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        if logger is not None: logger.store("train", EpRet=ep_ret, EpLen=ep_len)
                    ep_returns[idx] = 0
                    ep_lengths[idx] = 0
        rollout_end_time = time.time_ns()
        rollout_delta_time = (rollout_end_time - rollout_start_time) * 1e-9
        if logger is not None: logger.store("train", RolloutTime=rollout_delta_time, append=False)
