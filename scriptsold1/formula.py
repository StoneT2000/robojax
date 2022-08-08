import distrax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.env_util import make_vec_env

import walle_rl.agents.ppo
import walle_rl.explore as explore
from walle_rl.agents.ppo.agent import PPO
from walle_rl.agents.ppo.buffer import PPOBuffer
from walle_rl.architecture.ac.core import ActorCritic
from walle_rl.architecture.mlp import MLP
from walle_rl.architecture.model import Model
from walle_rl.common.random import PRNGSequence
from walle_rl.common.utils import get_action_dim
from walle_rl.logger.logger import Logger

# RNG sequence
rng = PRNGSequence(0)
np.random.seed(0)

env_id = "CartPole-v1"
num_cpu = 4
seed = 0
env = make_vec_env(env_id, num_cpu, seed=seed)
obs = env.reset()
act_dims = int(env.action_space.n)
actor = MLP([64, 64, act_dims], output_activation=None)
critic = MLP([64, 64, 1], output_activation=None)
ac = ActorCritic(
    rng=rng,
    actor=actor,
    critic=critic,
    explorer=explore.Categorical(),
    sample_obs=obs,
    act_dims=act_dims,
    actor_optim=optax.adam(learning_rate=1e-4),
    critic_optim=optax.adam(learning_rate=4e-4),
)
logger = Logger(
    tensorboard=False, wandb=False, cfg=dict(), workspace="workspace", exp_name="test"
)
steps_per_epoch = 2000
steps_per_epoch = steps_per_epoch // num_cpu
buffer = PPOBuffer(
    buffer_size=steps_per_epoch,
    observation_space=env.observation_space,
    action_space=env.action_space,
    n_envs=num_cpu,
)
algo = PPO(max_ep_len=200)

# a chain of mostly jaxxable components, we jit everything until something is not jittable?

# RL COMPONENTS

# Each one is a function taking as input some states
# (or wrapped states thathaveother functions in it like Model), some configurations, then outputting some other states

# update_parameters (actor: Model, critic: Model, buffer: Buffer, configs: ConfigState) -> new_actor_model, new_critic_model


chain = [
    # rollout with ac.actor and ac.critic -> Buffer
    # step function (can wrap env in it or not, depends on how env works)
    # configure rollout
    # - dapg (dataset state)
    # - intrinsic rewards (reward state?)
    # process_rollout Buffer -> Buffer
    # update_parameters
]

# updating a model training state with PPO
# ac.actor.params, ac.critic.params,
for c in chain:
    pass
