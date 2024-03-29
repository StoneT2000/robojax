from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium
import gymnasium.vector
import jax
from chex import Array
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium.wrappers import RecordVideo, TimeLimit
from omegaconf import OmegaConf

import robojax.utils.make_env._brax as _brax
import robojax.utils.make_env._dm_control as _dm_control
import robojax.utils.make_env._mani_skill2 as _mani_skill2


@dataclass
class EnvConfig:
    env_id: str
    jax_env: bool
    max_episode_steps: int
    num_envs: int
    env_kwargs: Dict


@dataclass
class EnvMeta:
    sample_obs: Array
    sample_acts: Array
    obs_space: spaces.Space
    act_space: spaces.Space


def make_env_from_cfg(cfg: EnvConfig, seed: int = None, video_path: str = None):
    if not isinstance(cfg.env_kwargs, dict):
        cfg.env_kwargs = OmegaConf.to_container(cfg.env_kwargs)
    return make_env(
        env_id=cfg.env_id,
        jax_env=cfg.jax_env,
        max_episode_steps=cfg.max_episode_steps,
        num_envs=cfg.num_envs,
        seed=seed,
        record_video_path=video_path,
        env_kwargs=cfg.env_kwargs,
    )


def make_env(
    env_id: str,
    jax_env: bool,
    max_episode_steps: int,
    num_envs: Optional[int] = 1,
    seed: Optional[int] = 0,
    record_video_path: str = None,
    env_kwargs=dict(),
    wrappers=[],
):
    """
    Utility function to create a jax/non-jax based environment given an env_id
    """
    if jax_env:
        import gymnax

        # from brax import envs
        from robojax.wrappers._gymnax import GymnaxToVectorGymWrapper, GymnaxWrapper

        if _brax.is_brax_env(env_id):
            env = _brax.env_factory(
                env_id=env_id,
                env_kwargs=env_kwargs,
                record_video_path=None,
                max_episode_steps=max_episode_steps,
            )()
        elif env_id in gymnax.registered_envs:
            env, env_params = gymnax.make(env_id)
            env = GymnaxWrapper(env, env_params, max_episode_steps=max_episode_steps, auto_reset=True)
        else:
            raise ValueError(f"Could not find environment {env_id} in gymnax or brax")
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None:
            print(f"Creating Jax-based env {env_id} as a normal VectorEnv for video recording")
            env = GymnaxToVectorGymWrapper(env, num_envs=num_envs)
            env = RecordVideo(env, video_folder=record_video_path, episode_trigger=lambda x: True)
            obs_space = env.single_observation_space
            act_space = env.single_action_space
            env.reset(seed=seed)
            sample_obs = obs_space.sample()
            sample_acts = act_space.sample()
        else:
            sample_acts = env.action_space().sample(jax.random.PRNGKey(0))
            obs_space = env.observation_space()
            sample_obs = obs_space.sample(jax.random.PRNGKey(0))
            act_space = env.action_space()
    else:

        def env_factory(env_id, idx, seed, record_video_path, env_kwargs, wrappers=[]):
            def _init():
                env = gymnasium.make(env_id, disable_env_checker=True, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                if record_video_path is not None and idx == 0:
                    env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True)
                return env

            return _init

        if _mani_skill2.is_mani_skill2_env(env_id):
            env_factory = _mani_skill2.env_factory

        elif _dm_control.is_dm_control_env(env_id):
            env_factory = _dm_control.env_factory

        wrappers.append(lambda x: TimeLimit(x, max_episode_steps=max_episode_steps))

        # create a vector env parallelized across CPUs with the given timelimit and auto-reset
        vector_env_cls = AsyncVectorEnv
        if num_envs == 1:
            vector_env_cls = SyncVectorEnv
        env: VectorEnv = vector_env_cls(
            [
                env_factory(
                    env_id,
                    idx,
                    seed=seed,
                    env_kwargs=env_kwargs,
                    record_video_path=record_video_path,
                    wrappers=wrappers,
                )
                for idx in range(num_envs)
            ]
        )
        obs_space = env.single_observation_space
        act_space = env.single_action_space
        env.reset(seed=seed)
        sample_obs = obs_space.sample()
        sample_acts = act_space.sample()

    return env, EnvMeta(
        obs_space=obs_space,
        act_space=act_space,
        sample_obs=sample_obs,
        sample_acts=sample_acts,
    )
