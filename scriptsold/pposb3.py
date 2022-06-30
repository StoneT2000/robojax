import gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
# env = gym.make("Pendulum-v0")

model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=10000*2)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()