from dataclasses import dataclass
from typing import Optional

import jax
from chex import Array
from gymnasium import spaces
import gymnasium as gym
import gymnasium.vector
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.wrappers import TimeLimit
@dataclass
class EnvMeta:
    sample_obs: Array
    sample_acts: Array
    obs_space: spaces.Space  # Technically not always the right typing
    act_space: spaces.Space


def make_env(env_id: str, jax_env: bool, max_episode_steps: int, num_envs: Optional[int] = 1, seed: Optional[int] = 0):
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
        try:
            import mani_skill2.envs
        except:
            print("Skipping ManiSkill2 import")
            pass
        # create a vector env parallelized across CPUs with the given timelimit and auto-reset
        env: VectorEnv = gym.vector.make(env_id, num_envs=num_envs, wrappers=[
            lambda x : TimeLimit(x, max_episode_steps=max_episode_steps)
        ])
        env.reset(seed=seed)
        obs_space = env.single_observation_space
        act_space = env.single_action_space
        sample_obs = obs_space.sample()
        sample_acts = act_space.sample()
    return env, EnvMeta(
        obs_space=obs_space,
        act_space=act_space,
        sample_obs=sample_obs,
        sample_acts=sample_acts,
    )
