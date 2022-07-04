import time
import distrax
import numpy as np
import walle_rl.agents.ppo
from walle_rl.agents.ppo.agent import PPO
from walle_rl.agents.ppo.buffer import PPOBuffer
from walle_rl.architecture.mlp import MLP
import gym

import flax.linen as nn
from walle_rl.architecture.model import Model
from walle_rl.common.random import PRNGSequence
import jax.numpy as jnp
import jax
from stable_baselines3.common.env_util import make_vec_env
from walle_rl.common.utils import get_action_dim
from walle_rl.architecture.ac.core import ActorCritic
import optax

import walle_rl.explore as explore
from walle_rl.logger.logger import Logger
# RNG sequence
rng = PRNGSequence(0)
np.random.seed(0)

env_id="CartPole-v1"
num_cpu = 4
seed = 0
env = make_vec_env(env_id, num_cpu, seed=seed)
obs = env.reset()
act_dims = int(env.action_space.n)
actor=MLP([64,64,act_dims], output_activation=None)
critic=MLP([64,64,1], output_activation=None)
ac = ActorCritic(
    rng=rng,
    actor=actor,
    critic=critic,
    explorer=explore.Categorical(),
    sample_obs=obs,
    act_dims=act_dims,
    actor_optim=optax.adam(learning_rate=1e-4),
    critic_optim=optax.adam(learning_rate=4e-4)
)
logger = Logger(tensorboard=False, wandb=False, cfg=dict(), workspace="workspace", exp_name="test")
steps_per_epoch = 2000
steps_per_epoch = steps_per_epoch // num_cpu
buffer = PPOBuffer(buffer_size=steps_per_epoch, observation_space=env.observation_space, action_space=env.action_space, n_envs=num_cpu)
algo = PPO(max_ep_len=500)
def t_cb(epoch):
    stats = logger.log(step=epoch)
    logger.pretty_print_table(stats)
    logger.reset()

stime = time.time_ns()
algo.train_loop(
    rng=rng,
    ac=ac,
    env=env,
    buffer=buffer,
    steps_per_epoch=steps_per_epoch,
    batch_size=512,
    logger=logger,
    update_iters=80,
    n_epochs=50,
    train_callback=t_cb,
)
etime = time.time_ns()
print(f"Time: {(etime-stime)*(1e-9)}")


# for i in range(1000):
#     a = ac.act(obs=obs, key=next(rng), deterministic=False)
#     env.render()
#     obs,_,_,_ = env.step(np.array(a))
