import gym
import torch

from paper_rl.modelfree.ppo import PPO
from paper_rl.architecture.ac.mlp import MLPActorCritic

env = gym.make("Pendulum-v1")
policy = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(64, 64))
# algo = PPO(
#     policy=None,
#     env=env
# )
obs = env.reset()
for i in range(1000):
    with torch.no_grad():
        action = policy.act(torch.tensor(obs), deterministic=True)
    print("a",action)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()