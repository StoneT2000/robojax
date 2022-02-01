import gym
import numpy as np
import torch
from paper_rl.logger.logger import Logger

from paper_rl.modelfree.ppo import PPO
from paper_rl.architecture.ac.mlp import MLPActorCritic
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.env_util import make_vec_env
# 2 ways to work with envs
# mpi it and have one env per process, each process has its own copy
# or use stable baselines vecenv style. - which also makes it easy to utilize the GPU.

if __name__ == "__main__":
    env_id = "Pendulum-v0"
    # env_id = "CartPole-v1"
    num_cpu = 4
    seed = 1
    # def make_env(gym_id, seed, idx):
    #     def thunk():
    #         env = gym.make(gym_id)
    #         if idx == 0:
    #             env = gym.wrappers.RecordVideo(env, f"videos/{0}")
    #         env.seed(seed)
    #         env.action_space.seed(seed)
    #         env.observation_space.seed(seed)
    #         return env
    #     return thunk
    # env = gym.vector.AsyncVectorEnv(
    #     [make_env(env_id, seed + i, i) for i in range(num_cpu)]
    # )
    env = make_vec_env(env_id, num_cpu, seed=seed)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(64, 64))
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    # vf_optimizer = torch.optim.Adam(model.v.parameters(), lr=3e-4)
    # pi_optimizer = torch.optim.Adam(model.pi.parameters(), lr=3e-4)

    # torch.set_num_threads(1)

    logger = Logger(tensorboard=True)
    steps_per_epoch=2000 // num_cpu
    batch_size=512
    algo = PPO(
        ac=model,
        env=env,
        num_envs=num_cpu,
        action_space=env.action_space,
        observation_space=env.observation_space,
        logger=logger,
        steps_per_epoch=steps_per_epoch,
        ent_coef=0.01,
        vf_coef=.5,
        train_iters=20,#80 // (steps_per_epoch * num_cpu // batch_size)
    )
    algo.train(max_ep_len=1000, n_epochs=200, 
    optim=optim,
    batch_size=batch_size)

    env.close()
    # eval_env = gym.vector.AsyncVectorEnv(
    #     [make_env(env_id, seed + i, i) for i in range(2)]
    # )
    eval_env = make_vec_env(env_id, 1, seed=seed)
    obs = eval_env.reset()
    
    for i in range(1000):
        with torch.no_grad():
            action = model.act(torch.tensor(obs), deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
        if done.any():
            print(info)
    eval_env.close()
    