import gym
import brax
from brax import envs
from brax.io import html, image
from IPython.display import HTML, Image 

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import optax
from stable_baselines3.common.env_util import make_vec_env

from robojax.agents.ppo.ppo import PPO
from robojax.models import explore
from robojax.models.ac.core import ActorCritic
from robojax.models.mlp import MLP

environment = "ant"
env = envs.create(env_name=environment, auto_reset=False, episode_length=1000)
state = env.reset(rng=jax.random.PRNGKey(0))

num_envs = 32
def env_step(rng_key, state, action):
    state = env.step(state, action * 0)
    return state.obs, state, state.reward, state.done != 0.0, state.info
def env_reset(rng_key):
    state = env.reset(rng_key)
    return state.obs, state
algo = PPO(env_step=env_step, env_reset=env_reset, jax_env=True)

act_dims = env.action_size
actor = MLP([256, 256, act_dims], output_activation=None)
critic = MLP([256, 256, 1], output_activation=None)
ac = ActorCritic(
    jax.random.PRNGKey(1),
    actor=actor,
    critic=critic,
    explorer=explore.Gaussian(act_dims=act_dims),
    sample_obs=env.reset(jax.random.PRNGKey(0)).obs,
    act_dims=act_dims,
    actor_optim=optax.adam(learning_rate=1e-4),
    critic_optim=optax.adam(learning_rate=4e-4),
)
algo.train(
    rng_key=jax.random.PRNGKey(0),
    steps_per_epoch=2048*4,
    update_iters=80,
    num_envs=num_envs,
    epochs=4000,
    ac=ac,
    batch_size=1024,
)
