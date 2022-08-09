import gym
import gymnax
import jax
import jax.numpy as jnp
import optax

from robojax.agents.ppo import PPO
from robojax.logger import Logger
from robojax.models import explore
from robojax.models.ac.core import ActorCritic
from robojax.models.mlp import MLP

env_id = "CartPole-v1"
num_envs = 4
env, env_params = gymnax.make("CartPole-v1")
algo = PPO(env_step=env.step, env_reset=env.reset, jax_env=True)

act_dims = int(env.action_space().n)
actor = MLP([64, 64, act_dims], output_activation=None)
critic = MLP([64, 64, 1], output_activation=None)
ac = ActorCritic(
    jax.random.PRNGKey(1),
    actor=actor,
    critic=critic,
    explorer=explore.Categorical(),
    sample_obs=env.reset(jax.random.PRNGKey(0))[0],
    act_dims=act_dims,
    actor_optim=optax.adam(learning_rate=1e-4),
    critic_optim=optax.adam(learning_rate=4e-4),
)
logger = Logger(
    tensorboard=True, wandb=True, cfg=dict(), workspace="robojax_exps", exp_name="ppo/cart_pole_gymnax", project_name="robojax"
)
algo.train(
    rng_key=jax.random.PRNGKey(0),
    steps_per_epoch=1000,
    update_iters=80,
    num_envs=num_envs,
    epochs=40,
    ac=ac,
    batch_size=512,
    logger=logger,
)
