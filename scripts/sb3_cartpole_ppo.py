import time
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1, n_steps=500, batch_size=500, n_epochs=20, ent_coef=0, target_kl=0.01, gae_lambda=0.97)
stime = time.time_ns()
model.learn(total_timesteps=2000*50)
etime = time.time_ns()
print(f"Time: {(etime-stime)*(1e-9)}")

model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()