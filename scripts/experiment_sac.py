import os
import os.path as osp
import warnings

import gym
from gym import spaces
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from flax import linen as nn
from stable_baselines3.common.env_util import make_vec_env

from robojax.agents.sac import SAC, ActorCritic, SACConfig
from robojax.agents.sac.networks import DiagGaussianActor, DoubleCritic
from robojax.cfg.parse import parse_cfg
from robojax.data.loop import GymLoop, JaxLoop
from robojax.logger import Logger
from robojax.models import explore
from robojax.models.mlp import MLP
from robojax.utils.spaces import get_action_dim

warnings.simplefilter(action="ignore", category=FutureWarning)


def main(cfg):
    env_id = cfg.env_id
    num_envs = 1
    sac_cfg = SACConfig(**cfg.sac)
    is_brax_env = False
    is_gymnax_env = False
    eval_loop = None
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
            sample_acts = env.action_space.sample(jax.random.PRNGKey(0))
            obs_space = env.observation_space
            act_space = env.action_space

            def seed_sampler(rng_key):
                return env.action_space(env_params).sample(rng_key)
        elif is_brax_env:
            env = envs.create(env_id, auto_reset=cfg.auto_reset)

            def env_step(rng_key, state, action):
                state = env.step(state, action)
                return (
                    state.obs,
                    state,
                    state.reward,
                    state.done != 0.0,
                    dict(**state.info, metrics=state.metrics),
                )

            def env_reset(rng_key):
                state = env.reset(rng_key)
                return state.obs, state

            def seed_sampler(rng_key):
                return jax.random.uniform(
                    rng_key, shape=(env.action_size, ), minval=-1.0, maxval=1.0, dtype=float)
            act_dims = env.action_size
            sample_obs = env.reset(jax.random.PRNGKey(0)).obs
            sample_acts = np.zeros(env.action_size)
            obs_space = spaces.Box(-np.ones(env.observation_size, float), np.ones(env.observation_size, float))
            act_space = spaces.Box(-np.ones(env.action_size, float), np.ones(env.action_size, float))
        algo = SAC(env_step=env_step, env_reset=env_reset, jax_env=cfg.jax_env,
                   observation_space=obs_space, action_space=act_space, seed_sampler=seed_sampler, cfg=sac_cfg)
        eval_loop = JaxLoop(env_reset=env_reset, env_step=env_step)
    else:
        env = gym.make(env_id)
        env = make_vec_env(env_id, num_envs, seed=cfg.seed)
        def seed_sampler(rng_key):
            return env.action_space.sample()[None, :]
        algo = SAC(env=env, jax_env=cfg.jax_env, observation_space=env.observation_space,seed_sampler=seed_sampler, action_space=env.action_space, cfg=sac_cfg)
        act_dims = get_action_dim(env.action_space)
        sample_obs = env.reset()
        sample_acts = env.action_space.sample()[None, :]
        eval_loop = GymLoop(env)

    assert env != None
    assert act_dims != None
    
    

    actor = DiagGaussianActor([256, 256], act_dims)
    critic = DoubleCritic([256, 256, 256, 256])
    ac = ActorCritic(
        jax.random.PRNGKey(cfg.seed),
        actor=actor,
        critic=critic,
        sample_obs=sample_obs,
        sample_acts=sample_acts,
        initial_temperature=sac_cfg.initial_temperature,
        actor_optim=optax.adam(learning_rate=cfg.model.actor_lr),
        critic_optim=optax.adam(learning_rate=cfg.model.critic_lr),
    )
    logger = Logger(
        cfg=cfg,
        **cfg.logger,
    )

    # def eval_apply(rng_key, params, obs):
    #     actor = params
    #     return actor(obs)[1], {}

    # best_ep_ret = 0

    # def train_callback(epoch, ac, rng_key, **kwargs):
    #     nonlocal best_ep_ret
    #     # every cfg.eval.eval_freq training epochs, evaluate our current model
    #     if epoch % cfg.eval.eval_freq == 0:
    #         rng_key, *eval_env_rng_keys = jax.random.split(
    #             rng_key, cfg.eval.num_eval_envs + 1
    #         )
    #         eval_buffer, _ = eval_loop.rollout(
    #             eval_env_rng_keys, ac.actor, eval_apply, cfg.eval.steps_per_env
    #         )
    #         eval_episode_ends = np.asarray(eval_buffer["done"])
    #         ep_rets = np.asarray(eval_buffer["ep_ret"])[
    #             eval_episode_ends].flatten()
    #         logger.store(
    #             tag="test",
    #             append=False,
    #             ep_ret=ep_rets,
    #             ep_len=np.asarray(eval_buffer["ep_len"])[
    #                 eval_episode_ends].flatten(),
    #         )
    #         ep_ret_avg = ep_rets.mean()
    #         if ep_ret_avg > best_ep_ret:
    #             best_ep_ret = ep_ret_avg
    #             ac.save(model_path)

    model_path = "weights.jx"  # osp.join(logger.exp_path, "weights.jx")
    # ac.load(model_path)

    algo.train(
        rng_key=jax.random.PRNGKey(cfg.seed),
        ac=ac,
        logger=logger,
        # train_callback=train_callback,
    )
    # ac.save(model_path)


if __name__ == "__main__":
    cfg = parse_cfg(
        default_cfg_path=osp.join(osp.dirname(__file__), "cfgs/sac/hopper.yml")
    )
    main(cfg)
