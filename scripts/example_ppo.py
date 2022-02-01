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
    env_id = "CartPole-v1"
    num_cpu = 1
    seed = 1
    env = make_vec_env(env_id, num_cpu, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(64, 64))
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    # vf_optimizer = torch.optim.Adam(model.v.parameters(), lr=3e-4)
    # pi_optimizer = torch.optim.Adam(model.pi.parameters(), lr=3e-4)

    # torch.set_num_threads(1)

    logger = Logger(tensorboard=True)
    steps_per_epoch=2048 // num_cpu
    batch_size=512
    algo = PPO(
        ac=model,
        env=env,
        num_envs=num_cpu,
        logger=logger,
        steps_per_epoch=steps_per_epoch,
        ent_coef=0.,
        vf_coef=.5,
        train_iters=10,#80 // (steps_per_epoch * num_cpu // batch_size)
    )
    algo.train(max_ep_len=1000, n_epochs=10, 
    optim=optim,
    batch_size=batch_size)

    eval_env = make_vec_env(env_id, 1)
    obs = eval_env.reset()
    for i in range(1000):
        with torch.no_grad():
            action = model.act(torch.tensor(obs), deterministic=True).reshape(1,)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
        if done.any():
            print(info)
        #   obs = env.reset()

    # env.close()