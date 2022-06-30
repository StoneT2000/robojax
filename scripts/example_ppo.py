
import walle_rl.agents.ppo
from walle_rl.architecture.mlp import MLP
import gym

import flax.linen as nn
from walle_rl.common.random import PRNGSequence
import jax.numpy as jnp
import jax
from stable_baselines3.common.env_util import make_vec_env
# RNG sequence
rng = PRNGSequence(0)

# define env
env_id="CartPole-v0"
num_cpu = 4
seed = 0
env = make_vec_env(env_id, num_cpu, seed=seed)
obs = env.reset()

# define model here, either your self or using our archs
model = MLP([64,64,64,2], output_activation=nn.tanh)
batch = jnp.ones((4,4))
variables = model.init(next(rng), obs)
output = model.apply(variables, batch)

# call training in a functional way



# Compose modules
"""
We need a model, e.g. MLP that does predicts actions

Then we need a exploration head


We need an algo, PPO, SAC, ... which can be deconstructed into
- optimization (DPG, clipped surrogate pg loss...)

We need an exploration strategy (sampler, intrinsic rewards) (which might be part of exploration head)

We need a environment

We need a experiment loop 
( which is own pipeline. E.g. rollout, learn, )

Run!

"""