import distrax
import walle_rl.agents.ppo
from walle_rl.architecture.mlp import MLP
import gym

import flax.linen as nn
from walle_rl.architecture.model import Model
from walle_rl.common.random import PRNGSequence
import jax.numpy as jnp
import jax
from stable_baselines3.common.env_util import make_vec_env

from walle_rl.architecture.ac.core import ActorCritic
import optax

import walle_rl.explore as explore
# RNG sequence
rng = PRNGSequence(0)

# define env
env_id="Pendulum-v1"
num_cpu = 4
seed = 0
env = make_vec_env(env_id, num_cpu, seed=seed)
obs = env.reset()
act_dims = 1
actor=MLP([64,64,64,1], output_activation=nn.tanh)
# , (4,), optax.adam(learning_rate=3e-4)
critic=MLP([64,64,64,1], output_activation=None)
# , next(rng), (4,), optax.adam(learning_rate=3e-4)
# explorer=Model.create(, next(rng), (), None)
ac = ActorCritic(
    rng=rng,
    actor=actor,
    critic=critic,
    explorer=explore.Gaussian(act_dims=act_dims),
    obs_shape=obs.shape,
    act_dims=act_dims,
    actor_optim=optax.adam(learning_rate=3e-4),
    critic_optim=optax.adam(learning_rate=3e-4)
)
# ac = Model.create(ac, next(rng), (4,))
for i in range(1000):
    a = ac.act(obs=obs, key=next(rng), deterministic=False)
    env.render()
    obs,_,_,_ = env.step(a)
