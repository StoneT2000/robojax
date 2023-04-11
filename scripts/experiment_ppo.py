import os.path as osp
import sys
import warnings

import jax
import optax
from flax import linen as nn

from robojax.agents.ppo import PPO
from robojax.agents.ppo.config import PPOConfig
from robojax.agents.ppo.networks import ActorCritic
from robojax.cfg.parse import parse_cfg
from robojax.models import explore
from robojax.models.mlp import MLP
from robojax.utils.make_env import make_env
from robojax.utils.spaces import get_action_dim

warnings.simplefilter(action="ignore", category=FutureWarning)


def main(cfg):
    # Setup the experiment parameters
    env_cfg = cfg.env
    cfg.eval_env = {**env_cfg, **cfg.eval_env}
    eval_env_cfg = cfg.eval_env

    video_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name, "videos")

    env, env_meta = make_env(
        env_id=env_cfg.env_id,
        jax_env=cfg.jax_env,
        max_episode_steps=env_cfg.max_episode_steps,
        num_envs=cfg.ppo.num_envs,
        seed=cfg.seed,
    )
    eval_env, _ = make_env(
        env_id=eval_env_cfg.env_id,
        jax_env=cfg.jax_env,
        max_episode_steps=eval_env_cfg.max_episode_steps,
        num_envs=cfg.ppo.num_eval_envs,
        seed=cfg.seed + 1000,
        record_video_path=video_path,
    )

    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts

    # create our actor critic models
    act_dims = get_action_dim(env_meta.act_space)
    # print("A",act_dims)
    # explorer=explore.Categorical()
    explorer = explore.Gaussian(act_dims=act_dims, log_std_scale=-0.5)

    # 64,64 for cartpole comparison
    # actor = MLP([64, 64, act_dims], output_activation=None)
    # critic = MLP([64, 64, 1], output_activation=None)
    # 32, 32, 32, 32 for brax
    # 256, 256, 256, 256, 256 for brax
    actor = MLP([32, 32, 32, 32, act_dims], output_activation=nn.tanh, final_ortho_scale=0.01)
    critic = MLP([256, 256, 256, 256, 256, 1], output_activation=None, final_ortho_scale=1)

    ac = ActorCritic.create(
        jax.random.PRNGKey(cfg.seed),
        actor=actor,
        critic=critic,
        explorer=explorer,
        sample_obs=sample_obs,
        sample_acts=sample_acts,
        actor_optim=optax.adam(learning_rate=cfg.model.actor_lr),
        critic_optim=optax.adam(learning_rate=cfg.model.critic_lr),
    )

    # create our algorithm
    ppo_cfg = PPOConfig(**cfg.ppo)
    algo = PPO(
        env=env,
        num_envs=cfg.ppo.num_envs,
        eval_env=eval_env,
        jax_env=cfg.jax_env,
        ac=ac,
        logger_cfg=dict(cfg=cfg, **cfg.logger),
        cfg=ppo_cfg,
    )

    model_path = osp.join(algo.logger.exp_path, "latest.jx")

    algo.train(rng_key=jax.random.PRNGKey(cfg.seed), steps=cfg.train.steps, verbose=1)
    ac.save(model_path)


if __name__ == "__main__":
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    main(cfg)
