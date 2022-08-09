import gym
import gymnax
import jax
import jax.numpy as jnp
from brax import envs
import optax
import os.path as osp
from robojax.agents.ppo import PPO
from robojax.agents.ppo.config import PPOConfig
from robojax.logger import Logger
from robojax.models import explore
from robojax.models.ac.core import ActorCritic
from robojax.models.mlp import MLP
from robojax.cfg.parse import parse_cfg
from stable_baselines3.common.env_util import make_vec_env
from robojax.utils.spaces import get_action_dim
def main(cfg):
    env_id = cfg.env_id
    num_envs = cfg.train.num_envs

    is_brax_env = False
    is_gymnax_env = False
    explorer = explore.Categorical()
    if cfg.jax_env:
        if env_id in gymnax.registered_envs:
            is_gymnax_env = True
        elif env_id in envs._envs:
            is_brax_env = True
        if is_gymnax_env:
            env, env_params = gymnax.make(env_id)
            env_step, env_reset = env.step, env.reset
            act_dims = int(env.action_space().n)
            sample_obs = env_reset(jax.random.PRNGKey(0))[0]
        elif is_brax_env:
            env = envs.create(env_id, auto_reset=False)
            def env_step(rng_key, state, action):
                state = env.step(state, action)
                return state.obs, state, state.reward, state.done != 0.0, dict(**state.info, metrics=state.metrics)
            def env_reset(rng_key):
                state = env.reset(rng_key)
                return state.obs, state
            act_dims = env.action_size
            explorer = explore.Gaussian(act_dims=act_dims, log_std_scale=-0.5)
            sample_obs = env.reset(jax.random.PRNGKey(0)).obs
        algo = PPO(env_step=env_step, env_reset=env_reset, jax_env=cfg.jax_env)
    else:
        env = gym.make(env_id)
        env = make_vec_env(env_id, num_envs, seed=cfg.seed)
        algo = PPO(env=env, jax_env=cfg.jax_env)
        act_dims = get_action_dim(env.action_space)
        sample_obs = env.reset()
    
    assert env != None
    assert act_dims != None

    algo.cfg = PPOConfig(**cfg.ppo)
    
    actor = MLP([256, 256, act_dims], output_activation=None)
    critic = MLP([256, 256, 1], output_activation=None)
    ac = ActorCritic(
        jax.random.PRNGKey(cfg.seed),
        actor=actor,
        critic=critic,
        explorer=explorer,
        sample_obs=sample_obs,
        act_dims=act_dims,
        actor_optim=optax.adam(learning_rate=cfg.model.actor_lr),
        critic_optim=optax.adam(learning_rate=cfg.model.critic_lr),
    )
    logger = Logger(
        cfg=cfg, **cfg.logger,
    )
    model_path = "weights.jx" #osp.join(logger.exp_path, "weights.jx")
    # ac.load(model_path)
    
    algo.train(
        rng_key=jax.random.PRNGKey(cfg.seed),
        steps_per_epoch=cfg.train.steps_per_epoch,
        update_iters=cfg.train.update_iters,
        num_envs=num_envs,
        epochs=cfg.train.epochs,
        ac=ac,
        batch_size=cfg.train.batch_size,
        logger=logger,
    )
    ac.save(model_path)




if __name__ == "__main__":
    cfg = parse_cfg(default_cfg_path=osp.join(osp.dirname(__file__), "cfgs/ant.yml"))
    print(cfg)
    main(cfg)