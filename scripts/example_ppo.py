import gym
import torch

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
    num_cpu = 2
    env = make_vec_env(env_id, num_cpu)
    # vec_env = make_vec_env(env_id, n_envs=num_cpu)
    o =env.reset()
    print(o)
    policy = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(64, 64))
    a = policy.act(torch.from_numpy(o))
    print(o.shape,a.shape)
    o, r, d, _ = env.step(a)
    env.num_envs
    print(o.shape, r, d, _)

    # env = make_env()
    
    # algo = PPO(
    #     policy=None,
    #     make_env=make_env
    # )
    # algo.train(
    #     batch_size=10000,
    #     steps_per_epoch=10000,

    # )


    # obs = env.reset()
    # for i in range(1000):
    #     with torch.no_grad():
    #         action = policy.act(torch.tensor(obs), deterministic=True)
    #     print("a",action)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #       obs = env.reset()

    # env.close()