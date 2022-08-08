import chex
from robojax.loop.loop import JaxLooper
import jax
import gymnax

from robojax.utils.random import PRNGSequence
env, env_params = gymnax.make("CartPole-v1")
print(env_params)

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    ep_ret: chex.Array
    reward: chex.Array
    ep_len: chex.Array
    done: chex.Array
rng = PRNGSequence(jax.random.PRNGKey(0))
def random_policy(rng_key, env_obs):
    return env.action_space(env_params).sample(rng_key), {}
def rollout_callback(
    action,
    env_obs,
    reward,
    ep_ret,
    ep_len,
    next_env_obs,
    done,
    **kwargs
):
    return TimeStep(action=action, obs=env_obs, ep_ret=ep_ret, ep_len=ep_len, reward=reward, done=done)
print("Roll out")
def env_step(key_step, state, action):
    return env.step(key_step, state=state, action=action, params=env_params,)
# o, s = env.reset(next(rng))
# env_step(next(rng), s, env.action_space(env_params).sample(next(rng)))
loop = JaxLooper(env.reset, env_step, apply_fn=random_policy)
import time

n_envs = 10
steps = 1_000_000 // n_envs
rng_key, *env_rng_keys = jax.random.split(jax.random.PRNGKey(0), n_envs+1)
# rollout_data = loop.rollout(env_rng_keys,steps, n_envs)
stime = time.time_ns()
rollout_data = loop.rollout(env_rng_keys,steps, n_envs)
# rollout_data = loop.rollout(jax.random.PRNGKey(0), env.reset, env_step, random_policy,steps, n_envs)
etime = time.time_ns()
rtime = (etime - stime) * 1e-9
FPS = steps*n_envs / (rtime)
print(f"FPS = {FPS}, {rtime}")
import ipdb;ipdb.set_trace()