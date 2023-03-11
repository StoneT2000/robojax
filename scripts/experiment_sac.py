import os
import os.path as osp
import time
import warnings

import jax
import numpy as np
import optax

from robojax.agents.sac import SAC, ActorCritic, SACConfig
from robojax.agents.sac.networks import DiagGaussianActor, DoubleCritic
from robojax.cfg.parse import parse_cfg
from robojax.utils.make_env import make_env
from robojax.utils.spaces import get_action_dim
from omegaconf import OmegaConf
warnings.simplefilter(action="ignore", category=FutureWarning)


def main(cfg):
    np.random.seed(cfg.seed)
    # Setup the experiment parameters
    env_cfg = cfg.env
    if "env_kwargs" not in env_cfg:
        env_cfg["env_kwargs"] = dict()
    cfg.eval_env = {**env_cfg, **cfg.eval_env}
    eval_env_cfg = cfg.eval_env

    video_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name, "videos")

    # create envs
    env, env_meta = make_env(env_id=env_cfg.env_id, jax_env=cfg.jax_env, max_episode_steps=env_cfg.max_episode_steps, num_envs=cfg.sac.num_envs, seed=cfg.seed, env_kwargs=OmegaConf.to_container(env_cfg.env_kwargs))
    eval_env = None
    if cfg.sac.num_eval_envs > 0:
        eval_env, _ = make_env(
            env_id=eval_env_cfg.env_id,
            jax_env=cfg.jax_env,
            max_episode_steps=eval_env_cfg.max_episode_steps,
            num_envs=cfg.sac.num_eval_envs,
            seed=cfg.seed + 1000,
            record_video_path=video_path,
            env_kwargs=OmegaConf.to_container(eval_env_cfg.env_kwargs)
        )
    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts

    # for SAC, we need a function to randomly sample actions, we write one for jax based and non jax based envs
    if cfg.jax_env:
        def seed_sampler(rng_key):
            return env.action_space().sample(rng_key)

    else:
        def seed_sampler(rng_key):
            return jax.random.uniform(
                rng_key,
                shape=(cfg.sac.num_envs, *env.single_action_space.shape),
                minval=-1.0,
                maxval=1.0,
                dtype=float,
            )

    # define hyperparameters for SAC
    sac_cfg = SACConfig(**cfg.sac)
    import dataclasses
    cfg.sac = dataclasses.asdict(sac_cfg)

    # create actor and critics models
    act_dims = get_action_dim(env_meta.act_space)
    actor = DiagGaussianActor(features=[256, 256, 256], act_dims=act_dims, state_dependent_std=True)
    critic = DoubleCritic(features=[256, 256, 256])
    
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

    # create our algorithm
    algo = SAC(
        env=env,
        eval_env=eval_env,
        num_envs=cfg.sac.num_envs,
        jax_env=cfg.jax_env,
        ac=ac,
        seed_sampler=seed_sampler,
        logger_cfg=dict(cfg=cfg, **cfg.logger),
        cfg=sac_cfg,
    )
    # algo.load_from_path("robojax_exps/maniskill2/PegInsertionSide/v4_s0/models/best_train_ep_ret_avg_ckpt.jx")
    # res = algo.evaluate(
    #     jax.random.PRNGKey(0),
    #     cfg.sac.num_eval_envs,
    #     1000,
    #     eval_loop=algo.eval_loop,
    #     params=ac.actor,
    #     apply_fn=algo.ac.act,
    # )
    # print(res)
    # print((res['eval_ep_lens'] < 200).mean())
    # exit()
    # train our algorithm with an initial seed
    algo.train(
        steps=cfg.train.steps,
        rng_key=jax.random.PRNGKey(cfg.seed),
    )


if __name__ == "__main__":
    cfg = parse_cfg(default_cfg_path=osp.join(osp.dirname(__file__), "cfgs/sac_peginsertion_stao.yml"))
    main(cfg)
