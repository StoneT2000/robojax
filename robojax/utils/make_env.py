from dataclasses import dataclass
from typing import Optional

import jax
from chex import Array
from gymnasium import spaces
import gymnasium
import gymnasium.vector
from gymnasium.vector import VectorEnv, AsyncVectorEnv
from gymnasium.wrappers import TimeLimit, RecordVideo
from robojax.wrappers.maniskill2 import ManiSkill2Wrapper, ContinuousTaskWrapper
import numpy as np
@dataclass
class EnvMeta:
    sample_obs: Array
    sample_acts: Array
    obs_space: spaces.Space  # Technically not always the right typing
    act_space: spaces.Space


def make_env(env_id: str, jax_env: bool, max_episode_steps: int, num_envs: Optional[int] = 1, seed: Optional[int] = 0, record_video_path: str = None):
    """
    Utility function to create a jax/non-jax based environment given an env_id
    """
    is_brax_env = False
    is_gymnax_env = False
    if jax_env:
        import gymnax
        from brax import envs

        from robojax.wrappers.brax import BraxGymWrapper

        if env_id in gymnax.registered_envs:
            is_gymnax_env = True
        elif env_id in envs._envs:
            is_brax_env = True
        else:
            raise ValueError(f"Could not find environment {env_id} in gymnax or brax")
        if is_gymnax_env:
            env, env_params = gymnax.make(env_id)
        elif is_brax_env:
            env = envs.create(env_id, auto_reset=True)
            # TODO make brax gym gymnasium compatible
            env = BraxGymWrapper(env)
        # TODO add time limit wrapper of sorts
        sample_obs = env.reset(jax.random.PRNGKey(0))[0]
        sample_acts = env.action_space().sample(jax.random.PRNGKey(0))
        obs_space = env.observation_space()
        act_space = env.action_space()
    else:
        wrappers = []
        
        mani_skill2_env = False
        try:
            import mani_skill2.envs
            from mani_skill2.utils.registration import REGISTERED_ENVS
            from mani_skill2.utils.wrappers import RecordEpisode
            gymnasium.register("LiftCube-v0", "mani_skill2.envs.pick_and_place.pick_cube:LiftCubeEnv")
            # wrappers.append(lambda x: RecordEpisode(x, output_dir="videos" + str(np.random.randint(0, 1000)), info_on_video=True))
            if env_id in REGISTERED_ENVS:
                mani_skill2_env = True
                wrappers.append(lambda x : ManiSkill2Wrapper(x))
                wrappers.append(lambda x : ContinuousTaskWrapper(x))
        except:
            print("Skipping ManiSkill2 import")
            pass
        wrappers.append(lambda x : TimeLimit(x, max_episode_steps=max_episode_steps))
        if mani_skill2_env:
            def make_env(env_id, idx, record_video):
                def _init():
                    env = gymnasium.make(env_id, disable_env_checker=True, control_mode="pd_ee_delta_pose")
                    if record_video and idx == 0:
                        env = RecordEpisode(env, record_video_path, info_on_video=True)
                    for wrapper in wrappers:
                        env = wrapper(env)
                    return env
                return _init
        # create a vector env parallelized across CPUs with the given timelimit and auto-reset
        # env: VectorEnv = gymnasium.vector.make(env_id, num_envs=num_envs, wrappers=wrappers, disable_env_checker=True)
        env: VectorEnv = AsyncVectorEnv([make_env(env_id, idx, record_video=record_video_path is not None) for idx in range(num_envs)])
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
