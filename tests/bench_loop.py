import time

import gymnax
import jax
import numpy as np
from gymnax.experimental import RolloutWrapper

from robojax.data import JaxLoop


def speed_gymnax_random(env_name, num_env_steps, num_envs, rng, env_kwargs):
    """Random episode rollout in gymnax."""
    # Define rollout manager for env
    if env_name == "Freeway-MinAtar":
        env_params = {"max_steps_in_episode": 1000}
    else:
        env_params = {}
    manager = RolloutWrapper(
        None, env_name=env_name, env_params=env_params, env_kwargs=env_kwargs
    )

    # Multiple rollouts for same network (different rng, e.g. eval)
    rng, rng_batch = jax.random.split(rng)
    rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
    if num_envs == 1:
        rollout_fn = manager.single_rollout
        obs, action, reward, next_obs, done, cum_ret = rollout_fn(rng_batch_eval, None)
        steps_per_batch = obs.shape[0]
    else:
        rollout_fn = manager.batch_rollout
        obs, action, reward, next_obs, done, cum_ret = rollout_fn(rng_batch_eval, None)
        steps_per_batch = obs.shape[0] * obs.shape[1]
    step_counter = 0

    start_t = time.time()
    # Loop over batch/single episode rollouts until steps are collected
    while step_counter < num_env_steps:
        rng, rng_batch = jax.random.split(rng)
        rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
        _ = rollout_fn(rng_batch_eval, None)
        step_counter += steps_per_batch
    return time.time() - start_t


def speed_gymnax_rollout_random(env_name, num_env_steps, num_envs, rng, env_kwargs):
    if env_name == "Freeway-MinAtar":
        env_params = {"max_steps_in_episode": 1000}
    else:
        env_params = {}

    env, _env_params = gymnax.make(env_name, **env_kwargs)
    env_params = _env_params.replace(**env_params)
    rng_key, *env_rng_keys = jax.random.split(rng, num_envs + 1)

    def env_step(key_step, state, action):
        return env.step(
            key_step,
            state=state,
            action=action,
            params=env_params,
        )

    def random_policy(rng_key, env_obs):
        return env.action_space(env_params).sample(rng_key), {}

    loop = JaxLoop(env.reset, env_step)
    # warmup rollout
    steps_per_env = num_env_steps // num_envs
    loop.rollout(env_rng_keys, random_policy, steps_per_env)
    start_t = time.time()
    loop.rollout(env_rng_keys, random_policy, steps_per_env)
    return time.time() - start_t


if __name__ == "__main__":

    # Benchmark gymnax way of rollouts, which isn't as fast since envs that are already done keep running still
    env_name = "CartPole-v1"

    num_runs = 3
    num_envs = 2
    total_env_steps = 1_000_000
    rng_key = jax.random.PRNGKey(0)
    r_times = []
    for run_id in range(num_runs):
        rng_key, rng_run_key = jax.random.split(rng_key)
        r_time = speed_gymnax_random(
            "CartPole-v1", total_env_steps, num_envs, rng_run_key, {}
        )
        r_times.append(r_time)
        print(f"Run {run_id + 1} - Env: {env_name} - Done after {r_time}")
    print(f"Avg: {np.mean(r_times)}, StdDev: {np.std(r_times)}")

    rng_key = jax.random.PRNGKey(0)
    for run_id in range(num_runs):
        rng_key, rng_run_key = jax.random.split(rng_key)
        r_time = speed_gymnax_rollout_random(
            "CartPole-v1", total_env_steps, num_envs, rng_run_key, {}
        )
        r_times.append(r_time)
        print(f"Run {run_id + 1} - Env: {env_name} - Done after {r_time}")
    print(f"Avg: {np.mean(r_times)}, StdDev: {np.std(r_times)}")
