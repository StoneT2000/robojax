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
):
    """
    Utility function to create a jax/non-jax based environment given an env_id
    """
    is_brax_env = False
    is_gymnax_env = False
    if jax_env:
        import gymnax
        from brax import envs

        from robojax.wrappers._brax import BraxGymWrapper
        from robojax.wrappers.gymnax import GymnaxWrapper

        if env_id in gymnax.registered_envs:
            is_gymnax_env = True
        elif env_id in envs._envs:
            is_brax_env = True
        else:
            raise ValueError(f"Could not find environment {env_id} in gymnax or brax")
        if is_gymnax_env:
            env, env_params = gymnax.make(env_id)
            env = GymnaxWrapper(env, env_params, max_episode_steps=max_episode_steps, auto_reset=True)
        elif is_brax_env:
            env = envs.create(env_id, episode_length=None, auto_reset=False)
            env = BraxGymWrapper(env, max_episode_steps=max_episode_steps, auto_reset=True)
        # sample_obs = #env.reset(jax.random.PRNGKey(0))[0]
        sample_acts = env.action_space().sample(jax.random.PRNGKey(0))
        obs_space = env.observation_space()
        sample_obs = obs_space.sample(jax.random.PRNGKey(0))
        act_space = env.action_space()
    else:

        def env_factory(env_id, idx, record_video, wrappers):
            def _init():
                env = gymnasium.make(env_id, disable_env_checker=True, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                if record_video and idx == 0:
                    env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True)
                return env

            return _init

        if _mani_skill2.is_mani_skill2_env(env_id):
            env_factory = _mani_skill2.env_factory(seed, env_kwargs, record_video_path)

        elif _dm_control.is_dm_control_env(env_id):
            env_factory = _dm_control.env_factory(seed, env_kwargs, record_video_path)

        wrappers = []
        wrappers.append(lambda x: TimeLimit(x, max_episode_steps=max_episode_steps))

        # create a vector env parallelized across CPUs with the given timelimit and auto-reset
        vector_env_cls = AsyncVectorEnv
        if num_envs == 1:
            vector_env_cls = SyncVectorEnv
        env: VectorEnv = vector_env_cls(
            [env_factory(env_id, idx, record_video=record_video_path is not None, wrappers=wrappers) for idx in range(num_envs)]
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
