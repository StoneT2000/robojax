import time
from functools import partial
from typing import Any, Callable, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from tqdm import tqdm

from robojax.agents.base import BasePolicy
from robojax.agents.ppo.config import PPOConfig, TimeStep
from robojax.agents.ppo.loss import (
    ActorAux,
    CriticAux,
    UpdateAux,
    actor_loss_fn,
    critic_loss_fn,
)
from robojax.agents.ppo.networks import ActorCritic, StepAux
from robojax.data.loop import EnvLoopState, GymLoop, JaxLoop
from robojax.data.sampler import BufferSampler
from robojax.models.model import Model
from robojax.utils import tools

PRNGKey = chex.PRNGKey

import pickle


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
class TrainStepMetrics:
    train_stats: Any
    train: Any
    time: Any
    update_aux: UpdateAux


@struct.dataclass
class PPOTrainState:
    ac: ActorCritic
    loop_state: EnvLoopState
    # rng
    rng_key: PRNGKey
    # monitoring
    total_env_steps: int
    """
    Total env steps sampled so far
    """
    training_steps: int
    """
    Total training steps so far
    """
    initialized: bool
    """
    When False, will automatically reset the loop state. This is usually false when starting training. When resuming training
    it will try to proceed from the previous loop state
    """


class PPO(BasePolicy):
    def __init__(
        self,
        jax_env: bool,
        ac: ActorCritic,
        env,
        num_envs,
        eval_env=None,
        num_eval_envs=1,
        logger_cfg=dict(),
        cfg: PPOConfig = {},
    ) -> None:
        super().__init__(jax_env, env, eval_env, num_envs, num_eval_envs, logger_cfg)
        if isinstance(cfg, dict):
            self.cfg = PPOConfig(**cfg)
        else:
            self.cfg = cfg

        self.state: PPOTrainState = PPOTrainState(
            ac=ac, rng_key=None, loop_state=EnvLoopState(), total_env_steps=0, training_steps=0, initialized=False
        )
        # self.ac: ActorCritic = ac

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
            self.collect_buffer = jax.jit(
                self.collect_buffer,
                static_argnames=["steps_per_env", "num_envs", "apply_fn"],
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

        rng_key, reset_rng_key = jax.random.split(rng_key, 2)

        # if env_obs is None, then this is the first time calling train and we prepare the environment
        if not self.state.initialized:
            loop_state = self.loop.reset_loop(reset_rng_key)
            self.state = self.state.replace(
                loop_state=loop_state,
                rng_key=rng_key,
                initialized=True,
            )

        start_step = self.state.total_env_steps

        if verbose:
            pbar = tqdm(total=steps + self.state.total_env_steps, initial=start_step)

        env_rollout_size = self.cfg.steps_per_env * self.cfg.num_envs

        def apply_fn(rng_key, ac: ActorCritic, obs):
            res = ac.step(rng_key, ac.actor, ac.critic, obs)
            return res

        while self.state.total_env_steps < start_step + steps:
            rng_key, train_rng_key = jax.random.split(self.state.rng_key)
            # TODO where should apply_fn go?
            state, train_step_metrics = self.train_step(train_rng_key, self.state, self.cfg, apply_fn)
            self.state: PPOTrainState = state.replace(rng_key=rng_key)

            # evaluate the current trained actor periodically
            if self.eval_loop is not None and tools.reached_freq(
                self.state.total_env_steps, self.cfg.eval_freq, step_size=env_rollout_size
            ):
                rng_key, eval_rng_key = jax.random.split(rng_key, 2)
                eval_results = self.evaluate(
                    eval_rng_key,
                    num_envs=self.cfg.num_eval_envs,
                    steps_per_env=self.cfg.eval_steps,
                    eval_loop=self.eval_loop,
                    params=state.ac.actor,
                    apply_fn=lambda rng_key, actor, obs: (
                        state.ac.act(rng_key, actor, obs, deterministic=True),
                        {},
                    ),
                )
                self.logger.store(
                    tag="test",
                    ep_ret=eval_results["eval_ep_rets"],
                    ep_len=eval_results["eval_ep_lens"],
                )
                self.logger.store(tag="test_stats", **eval_results["stats"])
                self.logger.log(self.state.total_env_steps)
                self.logger.reset()

            if verbose:
                pbar.update(n=env_rollout_size)
            if tools.reached_freq(self.state.total_env_steps, self.cfg.log_freq, env_rollout_size):
                self.logger.store(tag="train", **train_step_metrics.train)
                self.logger.store(tag="train_stats", **train_step_metrics.train_stats)
                total_time = time.time() - train_start_time  # TODO fix this to be consistent across train calls?

                self.logger.store(tag="time", **train_step_metrics.time)
                self.logger.store(
                    tag="time",
                    total=total_time,
                    SPS=state.total_env_steps / total_time,
                    total_env_steps=self.state.total_env_steps,
                    global_step=state.training_steps,  # TODO what does cleanrl log?
                )
                self.logger.log(state.total_env_steps)
                self.logger.reset()

    # this is completely jittable TODO
    def train_step(
        self,
        rng_key: PRNGKey,
        state: PPOTrainState,
        cfg: PPOConfig,
        apply_fn,
    ) -> Tuple[PPOTrainState, TrainStepMetrics]:
        rng_key, buffer_rng_key = jax.random.split(rng_key)
        rollout_s_time = time.time()

        # TODO can we prevent compilation here where init_env_states=None the first time for reset_env=False?

        # if we don't reset the environment after each rollout, then we generate environment states if we don't have any yet
        # for non jax envs this just resets the environments
        loop_state = state.loop_state
        if cfg.reset_env:
            # if self.last_env_states is None:
            rng_key, env_reset_rng_key = jax.random.split(rng_key, 2)
            loop_state = self.loop.reset_loop(env_reset_rng_key)
        loop_state, buffer = self.collect_buffer(
            rng_key=buffer_rng_key,
            loop_state=loop_state,
            steps_per_env=cfg.steps_per_env,
            num_envs=cfg.num_envs,
            ac=state.ac,
            apply_fn=apply_fn,
        )

        rollout_time = time.time() - rollout_s_time
        update_s_time = time.time()
        ac, update_aux = self.update_parameters(
            rng_key=rng_key, ac=state.ac, update_actor=True, update_critic=True, buffer=buffer
        )

        update_time = time.time() - update_s_time
        # TODO convert the dict below to a flax.struct.dataclass to improve speed
        state = state.replace(
            ac=ac,
            loop_state=loop_state,
            total_env_steps=state.total_env_steps + self.cfg.num_envs * self.cfg.steps_per_env,
            training_steps=state.training_steps + 1,
        )

        # compute some stats
        update_aux: UpdateAux
        num_eps = jnp.sum(buffer.done)
        train_metrics = dict(
            actor_loss=update_aux.actor_aux.actor_loss,
            approx_kl=update_aux.actor_aux.approx_kl,
            entropy=update_aux.actor_aux.entropy,
            critic_loss=update_aux.critic_aux.critic_loss,
            actor_updates=update_aux.actor_updates,
            critic_updates=update_aux.critic_updates,
        )
        if num_eps > 0:
            eps_dones = np.array(buffer.done.flatten())
            ep_ret = np.array(buffer.orig_ret.flatten())
            ep_len = np.array(buffer.ep_len.flatten())
            train_metrics["ep_ret"] = ep_ret[eps_dones]
            train_metrics["ep_len"] = ep_len[eps_dones]
        time_metrics = dict(
            update_time=update_time,
            rollout_time=rollout_time,
            rollout_fps=self.cfg.num_envs * self.cfg.steps_per_env / rollout_time,
        )
        return (state, TrainStepMetrics(train_stats=dict(), train=train_metrics, time=time_metrics, update_aux=update_aux))

    @partial(
        jax.jit,
        static_argnames=["self", "update_actor", "update_critic"],
    )
    def update_parameters(
        self,
        rng_key: PRNGKey,
        ac: ActorCritic,
        update_actor: bool,
        update_critic: bool,
        buffer: TimeStep,
    ) -> Tuple[ActorCritic, UpdateAux]:
        """Update the actor and critic parameters"""
        # TODO this is problematic if we pass entire buffer in if observations are very large
        buffer = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), buffer)  # (num_envs * steps_per_env, ...)
        sampler = BufferSampler(
            ["action", "env_obs", "log_p", "ep_ret", "adv"],
            buffer,
            buffer_size=buffer.action.shape[0],
            num_envs=self.cfg.num_envs,
        )
        # TODO improve speed? by having a while loop for the actor updates and a scan for critic updates
        def update_step_fn(_, data: Tuple[PRNGKey, Model, Model, bool, int, int, ActorAux]):
            (
                rng_key,
                actor,
                critic,
                can_update_actor,
                actor_updates,
                critic_updates,
                prev_actor_aux,
                prev_critic_aux,
            ) = data

            rng_key, sample_rng_key = jax.random.split(rng_key)
            # jax.debug.print("sample_rng_key: {}",sample_rng_key)
            batch = TimeStep(**sampler.sample_random_batch(sample_rng_key, self.cfg.batch_size))
            critic_aux = prev_critic_aux
            actor_aux = prev_actor_aux
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
                return actor, prev_actor_aux

            if update_actor:
                new_actor, actor_aux = jax.lax.cond(can_update_actor, update_actor_fn, skip_update_actor_fn, actor)
                actor_updates += 1 * can_update_actor
                can_update_actor = actor_aux.approx_kl <= self.cfg.target_kl * 1.5
            if update_critic:
                grads_c_fn = jax.grad(
                    critic_loss_fn(critic_apply_fn=critic.apply_fn, batch=batch),
                    has_aux=True,
                )
                grads, critic_aux = grads_c_fn(critic.params)
                new_critic = critic.apply_gradients(grads=grads)
                critic_updates += 1
            return (
                rng_key,
                new_actor,
                new_critic,
                can_update_actor,
                actor_updates,
                critic_updates,
                actor_aux,
                critic_aux,
            )

        update_init = (rng_key, ac.actor, ac.critic, update_actor, 0, 0, ActorAux(), CriticAux())
        carry = jax.lax.fori_loop(0, self.cfg.grad_updates_per_step, update_step_fn, update_init)

        _, actor, critic, _, actor_updates, critic_updates, actor_aux, critic_aux = carry
        ac = ac.replace(actor=actor, critic=critic)
        return (
            ac,
            UpdateAux(actor_aux=actor_aux, critic_aux=critic_aux, actor_updates=actor_updates, critic_updates=critic_updates),
        )

    def collect_buffer(
        self,
        rng_key,
        loop_state: EnvLoopState,
        steps_per_env: int,
        num_envs: int,
        ac: ActorCritic,
        apply_fn: Callable,
    ):
        # buffer collection is not jitted if env is not jittable

        # regardless this function returns a struct.dataclass object with all
        # the data in jax.numpy arrays for use
        # TODO make it so you only ever pass in one env rng key in, figure out batch dimension from loop state instead
        # for performance boost as splitting keys outside of jit is slow
        rng_key, *env_rng_keys = jax.random.split(rng_key, num_envs + 1)
        buffer, loop_state = self.loop.rollout(
            env_rng_keys,
            loop_state=loop_state,
            params=ac,
            apply_fn=apply_fn,
            steps_per_env=steps_per_env + 1,  # extra 1 for final value computation
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
        return loop_state, buffer

    def state_dict(self):
        ac = flax.serialization.to_bytes(self.state.ac)
        state_dict = dict(
            train_state=self.state.replace(ac=ac),
            logger=self.logger.state_dict(),
        )
        return state_dict

    def save(self, save_path: str):
        stime = time.time()
        state_dict = self.state_dict()
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saving Checkpoint {save_path}.", "Time:", time.time() - stime)

    def load_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        print(f"Loading Checkpoint {load_path}", state_dict["logger"])
        self.load(state_dict)

    def load(self, data):
        ac = flax.serialization.from_bytes(self.state.ac, data["train_state"].ac)
        # use serialized ac model
        self.state: PPOTrainState = data["train_state"].replace(ac=ac)
        # set initialized to False so previous env data is reset if it's not a jax env with env states we can start from
        if not self.jax_env:
            self.state = self.state.replace(initialized=False)
        self.logger.load(data["logger"])
