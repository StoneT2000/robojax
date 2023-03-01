import os
import os.path as osp
import warnings

import gym
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from flax import linen as nn
from stable_baselines3.common.env_util import make_vec_env

from robojax.agents.ppo import PPO
from robojax.agents.ppo.config import PPOConfig
from robojax.agents.ppo.networks import ActorCritic
from robojax.cfg.parse import parse_cfg
from robojax.data.loop import GymLoop, JaxLoop
from robojax.logger import Logger
from robojax.models import explore
from robojax.models.mlp import MLP
from robojax.utils.make_env import make_env
from robojax.utils.spaces import get_action_dim

warnings.simplefilter(action="ignore", category=FutureWarning)


def main(cfg):
    env_id = cfg.env_id
    
    env, env_meta = make_env(env_id, jax_env=cfg.jax_env, num_envs=cfg.ppo.num_envs, seed=cfg.seed)
    eval_env, _ = make_env(
        env_id,
        jax_env=cfg.jax_env,
        num_envs=cfg.ppo.num_eval_envs,
        seed=cfg.seed + 1000,
    )
    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts

    # create our actor critic models
    # act_dims = 2#sample_acts
    act_dims = get_action_dim(env_meta.act_space)
    print("A",act_dims)
    explorer=explore.Categorical()
    # explorer = explore.Gaussian(act_dims=act_dims, log_std_scale=-0.5)
    actor = MLP([256, 256, act_dims], output_activation=nn.tanh)
    critic = MLP([256, 256, 256, 256, 1], output_activation=None)
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

    # best_ep_ret = 0

    # def train_callback(epoch, ac, rng_key, **kwargs):
    #     nonlocal best_ep_ret
    #     # every cfg.eval.eval_freq training epochs, evaluate our current model
    #     if epoch % cfg.eval.eval_freq == 0:
    #         rng_key, *eval_env_rng_keys = jax.random.split(rng_key, cfg.eval.num_eval_envs + 1)
    #         eval_buffer, _ = eval_loop.rollout(eval_env_rng_keys, ac.actor, eval_apply, cfg.eval.steps_per_env)
    #         eval_episode_ends = np.asarray(eval_buffer["done"])
    #         ep_rets = np.asarray(eval_buffer["ep_ret"])[eval_episode_ends].flatten()
    #         logger.store(
    #             tag="test",
    #             append=False,
    #             ep_ret=ep_rets,
    #             ep_len=np.asarray(eval_buffer["ep_len"])[eval_episode_ends].flatten(),
    #         )
    #         ep_ret_avg = ep_rets.mean()
    #         if ep_ret_avg > best_ep_ret:
    #             best_ep_ret = ep_ret_avg
    #             ac.save(model_path)

    model_path = "weights.jx"  # osp.join(logger.exp_path, "weights.jx")
    # # ac.load(model_path)

    algo.train(
        rng_key=jax.random.PRNGKey(cfg.seed),
        epochs=cfg.train.epochs,
        verbose=1
    )
    ac.save(model_path)


if __name__ == "__main__":
    cfg = parse_cfg(default_cfg_path=osp.join(osp.dirname(__file__), "cfgs/ppo.yml"))
    main(cfg)
