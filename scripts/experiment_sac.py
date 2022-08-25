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
from robojax.utils.make_env import make_env
from robojax.utils.spaces import get_action_dim
from robojax.wrappers.brax import BraxGymWrapper

warnings.simplefilter(action="ignore", category=FutureWarning)


def main(cfg):
    np.random.seed(cfg.seed)
    env_id = cfg.env_id
    sac_cfg = SACConfig(**cfg.sac)
    env, env_meta = make_env(
        env_id, jax_env=cfg.jax_env, num_envs=cfg.sac.num_envs, seed=cfg.seed)
    eval_env, _ = make_env(
        env_id, jax_env=cfg.jax_env, num_envs=cfg.sac.num_eval_envs, seed=cfg.seed + 1000)
    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts
    if cfg.jax_env:
        def seed_sampler(rng_key):
            return env.action_space().sample(rng_key)[None, :]
    else:
        def seed_sampler(rng_key):
            return jax.random.uniform(rng_key, shape=env.action_space.shape, minval=-1.0, maxval=1.0, dtype=float)[None, :]

    algo = SAC(env=env, eval_env=eval_env, jax_env=cfg.jax_env, observation_space=env_meta.obs_space,
               action_space=env_meta.act_space, seed_sampler=seed_sampler, cfg=sac_cfg)
    act_dims = sample_acts.shape[0]

    actor = DiagGaussianActor([256, 256], act_dims)
    critic = DoubleCritic([256, 256])
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
    model_path = "weights.jx"  # osp.join(logger.exp_path, "weights.jx")
    # ac.load(model_path)

    algo.train(
        rng_key=jax.random.PRNGKey(cfg.seed),
        ac=ac,
        logger=logger,
    )
    # ac.save(model_path)


if __name__ == "__main__":
    import argparse
    import sys
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'base_config', metavar='int', type=int, choices=range(10),
    #      nargs='+', help='an integer in the range 0..9')
    cfg = parse_cfg(
        default_cfg_path=osp.join(osp.dirname(__file__), "cfgs/sac/halfcheetah_mujoco.yml")
    )
    main(cfg)
