import os
import os.path as osp
import time
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
from robojax.utils.make_env import make_env

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
            return env.action_space().sample(rng_key)
    else:
        def seed_sampler(rng_key):
            return jax.random.uniform(rng_key, shape=(cfg.sac.num_envs, *env.action_space.shape), minval=-1.0, maxval=1.0, dtype=float)

    if "exp_name" not in cfg.logger:
        cfg.logger["exp_name"] = f"{cfg.env_id}/sac/{round(time.time_ns() / 1000)}"
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
    algo = SAC(env=env, eval_env=eval_env, jax_env=cfg.jax_env, ac=ac, seed_sampler=seed_sampler, logger_cfg=dict(
        cfg=cfg,
        **cfg.logger
    ), cfg=sac_cfg)

    algo.load_from_path("robojax_exps/HalfCheetah-v3/sac/1661981/models/ckpt_100000.jx")


    algo.train(
        rng_key=jax.random.PRNGKey(cfg.seed),
    )


if __name__ == "__main__":
    import argparse
    import sys
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'base_config', metavar='int', type=int, choices=range(10),
    #      nargs='+', help='an integer in the range 0..9')
    cfg = parse_cfg(
        default_cfg_path=osp.join(osp.dirname(__file__), "cfgs/sac.yml")
    )
    main(cfg)
