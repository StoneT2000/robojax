import gym
import numpy as np
import torch
from paper_rl.architecture.ac.core import mlp
from paper_rl.logger.logger import Logger
import torch.nn as nn
from paper_rl.modelfree.gail import GAIL
from paper_rl.architecture.ac.mlp import MLPActorCritic

from stable_baselines3.common.env_util import make_vec_env
import os.path as osp

# 2 ways to work with envs
# mpi it and have one env per process, each process has its own copy
# or use stable baselines vecenv style. - which also makes it easy to utilize the GPU.

if __name__ == "__main__":
    # env_id = "Pendulum-v0"
    env_id = "CartPole-v1"
    num_cpu = 2
    seed = 1
    env = make_vec_env(env_id, num_cpu, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    gail_logger = Logger(tensorboard=True, exp_name="gail_test")
    gail_model = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(64, 64))
    class Discriminator(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden_sizes):
            super().__init__()
            self.mlp = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.Identity)

        def forward(self, observations, actions):
            if (len(actions.shape) == 1): actions = actions.unsqueeze(1)
            return torch.sigmoid(self.mlp(torch.hstack([observations, actions])))
    disc_model = Discriminator(env.observation_space.shape[0], 1, hidden_sizes=(64, 64))
    gail = GAIL(
        ac=gail_model,
        discriminator=disc_model,
        steps_per_epoch=2048 // num_cpu,
        env=env,
        n_envs=num_cpu,
        observation_space=env.observation_space,
        action_space=env.action_space,
        train_iters=10,
        logger=gail_logger,
    )

    pi_optimizer = torch.optim.Adam(gail_model.pi.parameters(), lr=1e-4)
    vf_optimizer = torch.optim.Adam(gail_model.v.parameters(), lr=3e-4)
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=1e-3)

    def train_callback(epoch, stats):
        filtered = {}
        for k in stats.keys():
            if (
                "Epoch" in k
                or "TotalEnvInteractions" in k
                or "EpRet" in k
                or "EpLen" in k
                # or "VVals" in k
                or "LossPi_avg" in k
                or "KL_avg" in k
                or "ClipFrac_avg" in k
                or "UpdateTime_avg" in k
                or "RolloutTime_avg" in k
                or "DiscriminatorLoss" in k
            ):
                filtered[k] = stats[k]
        gail_logger.pretty_print_table(filtered)
    def expert_reward(rewards, observations, actions):
        expert_rew_proportion = 1.#gail_hparam_args["expert_rew_proportion"]
        with torch.no_grad():
            observations = torch.as_tensor(observations, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            ps = disc_model.forward(observations, actions).squeeze(1).cpu().numpy()
            expert_rews = -np.log(ps) # negative log probs
            rews = expert_rews * expert_rew_proportion + rewards * (1 - expert_rew_proportion)
            return rews
    expert_trajectories = np.load(osp.join(osp.dirname(__file__), "expert_cartpole.npy"), allow_pickle=True)
    def sample_expert_trajectories(batch_size):
        obs = []
        acts = []
        for _ in range(batch_size):
            traj_id = np.random.randint(0, len(expert_trajectories))
            e_obs = expert_trajectories[traj_id]["observations"]
            e_acts = expert_trajectories[traj_id]["actions"]
            traj_step = np.random.randint(0, len(e_acts))
            obs.append(e_obs[traj_step])
            acts.append(e_acts[traj_step])
        return {
            "observations": torch.as_tensor(np.vstack(obs), dtype=torch.float32),
            "actions": torch.as_tensor(np.vstack(acts), dtype=torch.float32),
        }
    gail.train(
        train_callback=train_callback,
        rollout_callback=None,
        start_epoch=0,
        n_epochs=20,
        pi_optimizer=pi_optimizer,
        vf_optimizer=vf_optimizer,
        discrim_optimizer=disc_optimizer,
        ppo_batch_size=128,
        disc_mini_batch_size=128,
        expert_reward=expert_reward,
        sample_expert_trajectories=sample_expert_trajectories
    )

    eval_env = make_vec_env(env_id, 1, seed=seed)
    obs = eval_env.reset()

    for i in range(1000):
        with torch.no_grad():
            action = gail_model.act(torch.tensor(obs), deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
        if done.any():
            print(info)
    eval_env.close()


    # # for epoch in range(4):
    # # algo.train(max_ep_len=1000,start_epoch=epoch, n_epochs=1, optim=optim, batch_size=batch_size)
    # # algo.train(max_ep_len=1000, n_epochs=1, optim=optim, batch_size=batch_size)
    # # algo.train(max_ep_len=1000, n_epochs=1, optim=optim, batch_size=batch_size)
    # # algo.train(max_ep_len=1000, n_epochs=1, optim=optim, batch_size=batch_size)

    env.close()
    
