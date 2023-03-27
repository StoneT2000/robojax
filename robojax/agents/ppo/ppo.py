import time
from functools import partial
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from robojax.agents.base import BasePolicy
from robojax.agents.ppo.config import PPOConfig, TimeStep
from robojax.agents.ppo.loss import ActorAux, CriticAux, actor_loss_fn, critic_loss_fn
from robojax.agents.ppo.networks import ActorCritic, StepAux
from robojax.data.loop import GymLoop, JaxLoop, RolloutAux
from robojax.data.sampler import BufferSampler
from robojax.models.model import Model
from robojax.utils import tools

PRNGKey = chex.PRNGKey


@partial(jax.jit, static_argnames=["discount", "gae_lambda"])
@partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(rewards, dones, values, discount: float, gae_lambda: float):
    N = len(rewards)
    # values is of shape (N+1,)
    advantages = jnp.zeros((N + 1))
    not_dones = ~dones

    value_diffs = discount * values[1:] * not_dones - values[:-1]

    # in value_diffs we zero out whenever an episode was finished.
    # steps t where done = True, then values[1:][t] is zeroed (next_value at step t) as it is the value for the next episode
    deltas = rewards + value_diffs

    def body_fun(gae, t):
        gae = deltas[t] + discount * gae_lambda * not_dones[t] * gae
        return gae, gae

    indices = jnp.arange(N)[::-1]  # N - 1, N - 2, ..., 0
    _, advantages = jax.lax.scan(
        body_fun,
        0.0,
        indices,
    )

    advantages = advantages[::-1]
    return jax.lax.stop_gradient(advantages)


# TODO: Create a algo state / training state with separaable non-jax component (e.g. replay buffer) for easy saving and continuing runs


@struct.dataclass
class PPOTrainState:
    ac: ActorCritic


class PPO(BasePolicy):
    def __init__(
        self,
        jax_env: bool,
        ac: ActorCritic,
        env,
        num_envs,
        eval_env=None,
        logger_cfg=dict(),
        cfg: PPOConfig = {},
    ) -> None:
        super().__init__(jax_env, env, eval_env, num_envs, logger_cfg)
        if isinstance(cfg, dict):
            self.cfg = PPOConfig(**cfg)
        else:
            self.cfg = cfg

        self.step = 0
        self.last_env_states = None
        """
        When cfg.reset_env is False, we keep track of the last env states (obs, state, return, length) so we start the next rollout from there
        """
        self.ac: ActorCritic = ac

        # for jax or gym envs, define a custom rollout callback which collects the data we need for our replay buffer
        if self.jax_env:
            self.loop: JaxLoop

            def rollout_callback(
                action,
                env_obs,
                reward,
                ep_ret,
                ep_len,
                next_env_obs,
                terminated,
                truncated,
                info,
                aux: StepAux,
            ):
                done = terminated | truncated
                return TimeStep(
                    action=action,
                    env_obs=env_obs,
                    reward=reward,
                    adv=0,
                    log_p=aux.log_p,
                    ep_ret=ep_ret,
                    value=aux.value,
                    done=done,
                    ep_len=ep_len,
                    info=info,
                )

            self.loop.rollout_callback = rollout_callback
            self.loop.reset_env = self.cfg.reset_env
            self.collect_buffer = jax.jit(
                self.collect_buffer,
                static_argnames=["rollout_steps_per_env", "num_envs", "apply_fn"],
            )
        else:
            # we expect env to be a vectorized env now
            self.loop: GymLoop

            def rollout_callback(
                action,
                env_obs,
                reward,
                ep_ret,
                ep_len,
                next_env_obs,
                terminated,
                truncated,
                info,
                aux: StepAux,
            ):
                batch_size = len(env_obs)
                done = np.logical_or(terminated, truncated)
                return dict(
                    action=action,
                    env_obs=env_obs,
                    reward=reward,
                    adv=jnp.zeros((batch_size)),
                    log_p=aux.log_p,
                    ep_ret=ep_ret,
                    value=aux.value,
                    done=done,
                    ep_len=jnp.array(ep_len),
                )

            self.loop.rollout_callback = rollout_callback

    def train(
        self,
        rng_key: PRNGKey,
        steps: int,
        verbose: int = 1,
    ):
        """

        Args :
            steps : int
                Max number of environment samples before training is stopped.
        """

        train_start_time = time.time()

        def apply_fn(rng_key, params, obs):
            actor, critic = params
            res = self.ac.step(rng_key, actor, critic, obs)
            return res

        env_rollout_size = self.cfg.num_envs * self.cfg.steps_per_env
        # for t in range(epochs):
        total_env_steps = 0
        training_steps = 0
        while total_env_steps < steps:
            rng_key, train_rng_key = jax.random.split(rng_key)
            actor, critic, aux = self.train_step(
                rng_key=train_rng_key,
                update_iters=self.cfg.grad_updates_per_step,
                rollout_steps_per_env=self.cfg.steps_per_env,
                num_envs=self.cfg.num_envs,
                actor=self.ac.actor,
                critic=self.ac.critic,
                apply_fn=apply_fn,
                batch_size=self.cfg.batch_size,
            )
            self.ac.actor = actor
            self.ac.critic = critic
            total_env_steps += env_rollout_size
            training_steps += 1

            buffer = aux["buffer"]  # (T, num_envs)
            ep_lens = np.asarray(buffer.ep_len)
            ep_rets = np.asarray(buffer.orig_ret)
            ep_rews = np.asarray(buffer.reward)
            episode_ends = np.asarray(buffer.done)

            ### Logging ###
            if self.logger is not None:
                if self.eval_loop is not None and tools.reached_freq(
                    self.total_env_steps, self.cfg.eval_freq, step_size=env_rollout_size
                ):
                    rng_key, eval_rng_key = jax.random.split(rng_key, 2)
                    eval_results = self.evaluate(
                        eval_rng_key,
                        num_envs=self.cfg.num_eval_envs,
                        steps_per_env=self.cfg.eval_steps,
                        eval_loop=self.eval_loop,
                        params=self.ac.actor,
                        apply_fn=lambda rng_key, actor, obs: (
                            self.ac.act(rng_key, actor, obs, deterministic=True),
                            {},
                        ),
                    )
                    self.logger.store(
                        tag="test",
                        ep_ret=eval_results["eval_ep_rets"],
                        ep_len=eval_results["eval_ep_lens"],
                    )
                    self.logger.log(total_env_steps)
                    self.logger.reset()
                actor_updates = aux["update_aux"]["actor_updates"].item()
                actor_loss_aux: ActorAux = aux["update_aux"]["actor_loss_aux"]
                critic_loss_aux: CriticAux = aux["update_aux"]["critic_loss_aux"]
                total_time = time.time() - train_start_time
                if episode_ends.any():
                    self.logger.store(
                        tag="train",
                        ep_ret=ep_rets[episode_ends].flatten(),
                        ep_len=ep_lens[episode_ends].flatten(),
                    )
                self.logger.store(
                    tag="train",
                    ep_rew=ep_rews.flatten(),
                    fps=env_rollout_size / aux["rollout_time"],
                    env_steps=self.total_env_steps,
                    # note we slice after moving to numpy arrays for slight performance boost
                    # as slicing jax arrays creates a call to compile a new dynamic_slice
                    entropy=np.asarray(actor_loss_aux.entropy)[:actor_updates],
                    actor_loss=np.asarray(actor_loss_aux.actor_loss)[:actor_updates],
                    approx_kl=np.asarray(actor_loss_aux.approx_kl)[:actor_updates],
                    critic_loss=np.asarray(critic_loss_aux.critic_loss),
                    actor_updates=actor_updates,
                    critic_updates=aux["update_aux"]["critic_updates"].item(),
                )

                self.logger.store(
                    tag="time",
                    rollout=aux["rollout_time"],
                    update=aux["update_time"],
                    total=total_time,
                    sps=total_env_steps / total_time,
                    epoch=training_steps,
                )
                self.logger.log(total_env_steps)
                self.logger.reset()

            self.step += 1

    def train_step(
        self,
        rng_key: PRNGKey,
        update_iters: int,
        rollout_steps_per_env: int,
        num_envs: int,
        actor: Model,
        critic: Model,
        apply_fn: Callable,
        batch_size: int,
    ):
        rng_key, buffer_rng_key = jax.random.split(rng_key)
        rollout_s_time = time.time()

        # TODO can we prevent compilation here where init_env_states=None the first time for reset_env=False?

        # if we don't reset the environment after each rollout, then we generate environment states if we don't have any yet
        # for non jax envs this just resets the environments
        if not self.cfg.reset_env and self.jax_env:
            if self.last_env_states is None:
                rng_key, env_reset_rng_key = jax.random.split(rng_key, num_envs)
                self.last_env_states = self.loop.reset_loop(env_reset_rng_key)
        buffer, info = self.collect_buffer(
            rng_key=buffer_rng_key,
            rollout_steps_per_env=rollout_steps_per_env,
            num_envs=num_envs,
            actor=actor,
            critic=critic,
            apply_fn=apply_fn,
            init_env_states=self.last_env_states,
        )

        if not self.cfg.reset_env:
            info: RolloutAux
            self.last_env_states = (
                info.final_env_obs,
                info.final_env_state,
                info.final_ep_returns,
                info.final_ep_lengths,
            )

        rollout_time = time.time() - rollout_s_time
        update_s_time = time.time()
        actor, critic, update_aux = self.update_parameters(
            rng_key=rng_key,
            actor=actor,
            critic=critic,
            update_actor=True,
            update_critic=True,
            update_iters=update_iters,
            num_envs=num_envs,
            batch_size=batch_size,
            buffer=buffer,
        )

        update_time = time.time() - update_s_time
        # TODO convert the dict below to a flax.struct.dataclass to improve speed
        return (
            actor,
            critic,
            dict(
                buffer=buffer,
                update_aux=update_aux,
                rollout_time=rollout_time,
                update_time=update_time,
            ),
        )

    @partial(
        jax.jit,
        static_argnames=[
            "self",
            "update_actor",
            "update_critic",
            "update_iters",
            "num_envs",
            "batch_size",
        ],
    )
    def update_parameters(
        self,
        rng_key: PRNGKey,
        actor: Model,
        critic: Model,
        update_actor: bool,
        update_critic: bool,
        update_iters: int,
        num_envs: int,
        batch_size: int,
        buffer: TimeStep,
    ) -> Tuple[Model, Model, Dict]:
        """Update the actor and critic parameters"""
        sampler = BufferSampler(
            ["action", "env_obs", "log_p", "ep_ret", "adv"],
            buffer,
            buffer_size=buffer.action.shape[0],
            num_envs=num_envs,
        )
        # TODO improve speed? by having a while loop for the actor updates and a scan for critic updates
        def update_step_fn(data: Tuple[PRNGKey, Model, Model, bool, int, int, ActorAux], _):
            (
                rng_key,
                actor,
                critic,
                can_update_actor,
                actor_updates,
                critic_updates,
                prev_actor_loss_aux,
            ) = data
            rng_key, sample_rng_key = jax.random.split(rng_key)
            batch = TimeStep(**sampler.sample_random_batch(sample_rng_key, batch_size))
            info_c = None
            new_actor = actor
            new_critic = critic

            def update_actor_fn(actor):
                grads_a_fn = jax.grad(
                    actor_loss_fn(
                        clip_ratio=self.cfg.clip_ratio,
                        entropy_coef=self.cfg.ent_coef,
                        actor_apply_fn=actor.apply_fn,
                        batch=batch,
                    ),
                    has_aux=True,
                )
                grads, info_a = grads_a_fn(actor.params)
                new_actor = actor.apply_gradients(grads=grads)

                return new_actor, info_a

            def skip_update_actor_fn(actor):
                return actor, prev_actor_loss_aux

            if update_actor:
                new_actor, actor_loss_aux = jax.lax.cond(can_update_actor, update_actor_fn, skip_update_actor_fn, actor)
                actor_updates += 1 * can_update_actor
                can_update_actor = jax.lax.cond(
                    actor_loss_aux.approx_kl > self.cfg.target_kl * 1.5,
                    lambda: False,
                    lambda: True,
                )
            else:
                actor_loss_aux = prev_actor_loss_aux
            if update_critic:
                grads_c_fn = jax.grad(
                    critic_loss_fn(critic_apply_fn=critic.apply_fn, batch=batch),
                    has_aux=True,
                )
                grads, info_c = grads_c_fn(critic.params)
                new_critic = critic.apply_gradients(grads=grads)
                critic_updates += 1
            return (
                rng_key,
                new_actor,
                new_critic,
                can_update_actor,
                actor_updates,
                critic_updates,
                actor_loss_aux,
            ), dict(actor_loss_aux=actor_loss_aux, critic_loss_aux=info_c)

        update_init = (rng_key, actor, critic, update_actor, 0, 0, ActorAux())
        carry, update_aux = jax.lax.scan(update_step_fn, update_init, (), length=update_iters)

        _, actor, critic, _, actor_updates, critic_updates, _ = carry

        return (
            actor,
            critic,
            dict(**update_aux, actor_updates=actor_updates, critic_updates=critic_updates),
        )

    def collect_buffer(
        self,
        rng_key,
        rollout_steps_per_env: int,
        num_envs: int,
        actor,
        critic,
        apply_fn: Callable,
        init_env_states=None,
    ):
        # buffer collection is not jitted if env is not jittable

        # regardless this function returns a struct.dataclass object with all
        # the data in jax.numpy arrays for use
        rng_key, *env_rng_keys = jax.random.split(rng_key, num_envs + 1)
        buffer, aux = self.loop.rollout(
            env_rng_keys,
            params=(actor, critic),
            apply_fn=apply_fn,
            steps_per_env=rollout_steps_per_env + 1,  # extra 1 for final value computation
            init_env_states=init_env_states,
        )
        buffer: TimeStep

        if not self.jax_env:
            # if not a jax based env, then buffer is a python dictionary and we
            # convert it
            buffer = TimeStep(**buffer)
        # rest of the code here is jitted and / or vmapped
        advantages = gae_advantages(
            buffer.reward[:-1],
            buffer.done[:-1],
            buffer.value,
            self.cfg.discount,
            self.cfg.gae_lambda,
        )
        returns = advantages + buffer.value[-1, :]
        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # TODO can we speed up this replace op?
        # exclude last step which was used just for bootstrapping. e.g. https://github.com/deepmind/acme/blob/master/acme/agents/jax/ppo/learning.py#L331
        buffer = buffer.replace(
            adv=advantages,
            ep_ret=returns,
            reward=buffer.reward[:-1],
            orig_ret=buffer.ep_ret[:-1],
            log_p=buffer.log_p[:-1],
            action=buffer.action[:-1],
            env_obs=buffer.env_obs[:-1],
            done=buffer.done[:-1],
            ep_len=buffer.ep_len[:-1],
        )
        return buffer, aux

    @property
    def total_env_steps(self):
        env_steps_per_epoch = self.cfg.num_envs * self.cfg.steps_per_env
        return self.step * env_steps_per_epoch

    def state_dict(self):
        state_dict = dict(ac=self.ac.state_dict(), step=self.step, logger=self.logger.state_dict())
        return state_dict

    def load(self, data):
        self.ac = self.ac.load(data["ac"])
        self.step = data["step"]
        self.logger.load(data["logger"])
        return self
