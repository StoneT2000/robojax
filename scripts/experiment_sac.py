import os.path as osp
import sys
import warnings

import jax
import numpy as np
import optax
from omegaconf import OmegaConf

from robojax.agents.sac import SAC, ActorCritic, SACConfig
from robojax.agents.sac.networks import DiagGaussianActor, DoubleCritic
from robojax.cfg.parse import parse_cfg
from robojax.logger import LoggerConfig
from robojax.models import NetworkConfig, build_network_from_cfg
from robojax.utils.make_env import EnvConfig, make_env_from_cfg
from robojax.utils.spaces import get_action_dim

warnings.simplefilter(action="ignore", category=FutureWarning)

from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    steps: int
    actor_lr: float
    critic_lr: float


@dataclass
class SACNetworkConfig:
    actor: NetworkConfig
    critic: NetworkConfig


@dataclass
class SACExperiment:
    seed: int
    sac: SACConfig
    env: EnvConfig
    eval_env: EnvConfig
    train: TrainConfig
    network: SACNetworkConfig
    logger: LoggerConfig
    algo: str = "sac"


from dacite import from_dict


def main(cfg: SACExperiment):
    np.random.seed(cfg.seed)
    ### Setup the experiment parameters ###

    # Setup training and evaluation environment configs
    env_cfg = cfg.env
    if "env_kwargs" not in env_cfg:
        env_cfg["env_kwargs"] = dict()
    cfg.eval_env = {**env_cfg, **cfg.eval_env}
    cfg = from_dict(data_class=SACExperiment, data=OmegaConf.to_container(cfg))
    eval_env_cfg = cfg.eval_env

    video_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name, "videos")

    cfg.sac.num_envs = cfg.env.num_envs
    cfg.sac.num_eval_envs = cfg.eval_env.num_envs

    # create envs
    env, env_meta = make_env_from_cfg(env_cfg, seed=cfg.seed)
    eval_env = None
    if cfg.sac.num_eval_envs > 0:
        eval_env, _ = make_env_from_cfg(eval_env_cfg, seed=cfg.seed + 1_000_000, video_path=video_path)
    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts

    # for SAC, we need a function to randomly sample actions, we write one for jax based and non jax based envs
    if cfg.env.jax_env:

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

    # create actor and critics models
    act_dims = get_action_dim(env_meta.act_space)
    actor = DiagGaussianActor(
        feature_extractor=build_network_from_cfg(cfg.network.actor), act_dims=act_dims, state_dependent_std=True
    )
    critic = DoubleCritic(feature_extractor=build_network_from_cfg(cfg.network.critic))
    ac = ActorCritic.create(
        jax.random.PRNGKey(cfg.seed),
        actor=actor,
        critic=critic,
        sample_obs=sample_obs,
        sample_acts=sample_acts,
        initial_temperature=cfg.sac.initial_temperature,
        actor_optim=optax.adam(learning_rate=cfg.train.actor_lr),
        critic_optim=optax.adam(learning_rate=cfg.train.critic_lr),
    )

    # create our algorithm
    algo = SAC(
        env=env,
        eval_env=eval_env,
        jax_env=cfg.env.jax_env,
        ac=ac,
        seed_sampler=seed_sampler,
        logger_cfg=dict(cfg=cfg, **asdict(cfg.logger)),
        cfg=SACConfig(**asdict(cfg.sac)),
    )

    # train our algorithm with an initial seed
    algo.train(steps=cfg.train.steps, rng_key=jax.random.PRNGKey(cfg.seed))


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    del sys.argv[1]
    cfg = parse_cfg(default_cfg_path=cfg_path)
    main(cfg)
