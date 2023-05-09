"""
python bench_loop.py compare
"""
import time

import gymnax
import jax
import numpy as np
from gymnax.experimental import RolloutWrapper

from robojax.data import JaxLoop
from robojax.wrappers.gymnax import GymnaxWrapper


def speed_gymnax_random(env_name, num_env_steps, num_envs, rng, env_kwargs, max_episode_steps):
    """Random episode rollout in gymnax."""
    # Define rollout manager for env
    if env_name == "Freeway-MinAtar":
        env_params = {"max_steps_in_episode": 1000}
    else:
        env_params = {"max_steps_in_episode": max_episode_steps}
    manager = RolloutWrapper(
        None, env_name=env_name, env_params=env_params, env_kwargs=env_kwargs, num_env_steps=max_episode_steps
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

    iterations = num_env_steps // steps_per_batch
    # Loop over batch/single episode rollouts until steps are collected
    # @jax.jit
    def run(rng, iterations):
        def body_fn(data, _):
            rng = data
            rng, rng_batch = jax.random.split(rng)
            rng_batch_eval = jax.random.split(rng_batch, num_envs).squeeze()
            out = rollout_fn(rng_batch_eval, None)
            # step_counter += steps_per_batch
            return rng, out

        _, out = jax.lax.scan(body_fn, (rng), None, iterations)
        # while step_counter < num_env_steps:
        return out

    # warmup
    jax.jit(run, static_argnames=["iterations"])(rng, iterations)
    for i in range(3):
        start_t = time.time()
        out = run(rng, iterations)
        jax.block_until_ready(out)
        delta = time.time() - start_t
        print(delta)
    return delta


def speed_gymnax_rollout_random(env_name, num_env_steps, num_envs, rng, env_kwargs, max_episode_steps):
    if env_name == "Freeway-MinAtar":
        env_params = {"max_steps_in_episode": 1000}
    else:
        env_params = {}

    env, _env_params = gymnax.make(env_name, **env_kwargs)

    env_params = _env_params.replace(**env_params)
    env = GymnaxWrapper(env, _env_params, max_episode_steps=max_episode_steps)
    rng_key, reset_rng_key, *env_rng_keys = jax.random.split(rng, num_envs + 2)

    def random_policy(rng_key, params, env_obs):
        return env.action_space(env_params).sample(rng_key), {}

    loop = JaxLoop(env.reset, env.step, num_envs=num_envs)
    loop_state = loop.reset_loop(reset_rng_key)
    # warmup rollout

    steps_per_env = num_env_steps // num_envs
    print(steps_per_env)
    out = loop.rollout(env_rng_keys, loop_state, None, random_policy, steps_per_env)
    for _ in range(3):
        start_t = time.time()
        loop_state = loop.reset_loop(reset_rng_key)
        out = loop.rollout(env_rng_keys, loop_state, None, random_policy, steps_per_env)
        jax.block_until_ready(out)
        delta = time.time() - start_t
        print(delta)
    return delta


if __name__ == "__main__":
    # Benchmark gymnax way of rollouts, which isn't as fast since envs that are already done keep running still
    env_name = "CartPole-v1"
    max_episode_steps = 500
    num_runs = 3
    num_envs = 4000
    total_env_steps = 10_000_000
    rng_key = jax.random.PRNGKey(0)
    r_times = []
    for run_id in range(num_runs):
        rng_key, rng_run_key = jax.random.split(rng_key)
        r_time = speed_gymnax_random("CartPole-v1", total_env_steps, num_envs, rng_run_key, {}, max_episode_steps)
        r_times.append(r_time)
        print(f"Run {run_id + 1} - Env: {env_name} - Done after {r_time}")
    print(f"Avg: {np.mean(r_times)}, StdDev: {np.std(r_times)}")

    r_times = []
    rng_key = jax.random.PRNGKey(0)
    for run_id in range(num_runs):
        rng_key, rng_run_key = jax.random.split(rng_key)
        r_time = speed_gymnax_rollout_random("CartPole-v1", total_env_steps, num_envs, rng_run_key, {}, max_episode_steps)
        r_times.append(r_time)
        print(f"Run {run_id + 1} - Env: {env_name} - Done after {r_time}")
    print(f"Avg: {np.mean(r_times)}, StdDev: {np.std(r_times)}")
