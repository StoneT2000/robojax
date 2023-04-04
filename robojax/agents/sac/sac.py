import os
import pickle
import time
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple

import distrax
import flax
import jax
import numpy as np
from chex import Array, PRNGKey
from flax import struct
from tqdm import tqdm

from robojax.agents.base import BasePolicy
from robojax.agents.sac import loss
from robojax.agents.sac.config import SACConfig, TimeStep
from robojax.agents.sac.networks import ActorCritic, DiagGaussianActor
from robojax.data.buffer import GenericBuffer
from robojax.data.loop import EnvAction
from robojax.utils import tools


@struct.dataclass
class TrainStepMetrics:
    train_stats: Any
    train: Any
    time: Any


@struct.dataclass
class SACTrainState:
    # model states
    ac: ActorCritic

    # env states
    env_obs: Array
    env_states: Array
    ep_lens: Array
    ep_rets: Array
    dones: Array

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


class SAC(BasePolicy):
    def __init__(
        self,
        jax_env: bool,
        ac: ActorCritic,
        num_envs,
        env,
        seed_sampler: Callable[[PRNGKey], EnvAction] = None,
        eval_env=None,
        logger_cfg=dict(),
        cfg: SACConfig = {},
    ):
        super().__init__(jax_env, env, eval_env, num_envs, logger_cfg)
        if isinstance(cfg, dict):
            self.cfg = SACConfig(**cfg)
        else:
            self.cfg = cfg

        self.state: SACTrainState = SACTrainState(
            ac=ac,
            env_obs=None,
            env_states=None,
            ep_lens=None,
            ep_rets=None,
            dones=None,
            total_env_steps=0,
            training_steps=0,
            rng_key=None,
            initialized=False,
        )

        if seed_sampler is None:
            seed_sampler = lambda rng_key: self.env.action_space().sample(rng_key)
            # TODO add a nice error message if this guessed sampler doesn't work
        self.seed_sampler = seed_sampler

        buffer_config = dict(
            action=((self.action_dim,), self.action_space.dtype),
            reward=((), np.float32),
            mask=((), float),
        )
        if isinstance(self.obs_shape, dict):
            buffer_config["env_obs"] = (
                self.obs_shape,
                {k: self.observation_space[k].dtype for k in self.observation_space},
            )
        else:
            buffer_config["env_obs"] = (self.obs_shape, np.float32)
        buffer_config["next_env_obs"] = buffer_config["env_obs"]

        # note that we use GenericBuffer class, which is not backed by jax for storing
        # interactions due to relative slow overhead for adding small amounts of data and moving data
        # off the GPU
        self.replay_buffer = GenericBuffer(
            buffer_size=self.cfg.replay_buffer_capacity,
            num_envs=self.cfg.num_envs,
            config=buffer_config,
        )

        if self.cfg.target_entropy is None:
            self.cfg.target_entropy = -self.action_dim / 2

        # Jax env specific code to improve speed
        if self.jax_env:
            self._env_step = jax.jit(self._env_step, static_argnames=["seed"])

    @partial(jax.jit, static_argnames=["self", "seed"])
    def _sample_action(self, rng_key, actor: DiagGaussianActor, env_obs, seed=False):
        if seed:
            a = self.seed_sampler(rng_key)
        else:
            dist: distrax.Distribution = actor(env_obs)
            a = dist.sample(seed=rng_key)
        return a

    def _env_step(self, rng_key: PRNGKey, env_obs, env_state, actor: DiagGaussianActor, seed=False):
        rng_key, act_rng_key, env_rng_key = jax.random.split(rng_key, 3)
        a = self._sample_action(act_rng_key, actor, env_obs, seed)
        if self.jax_env:
            (
                next_env_obs,
                next_env_state,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_step(env_rng_key, env_state, a)
        else:
            a = tools.any_to_numpy(a)
            next_env_obs, reward, terminated, truncated, info = self.env.step(a)
            next_env_state = None
        return a, next_env_obs, next_env_state, reward, terminated, truncated, info

    def train(self, rng_key: PRNGKey, steps: int, verbose=1):
        """
        Args :
            rng_key: PRNGKey,
                Random key to seed the training with. It is only used if train() was never called before, otherwise the code uses self.state.rng_key
            steps : int
                Max number of environment samples before training is stopped.
        """
        train_start_time = time.time()

        rng_key, reset_rng_key = jax.random.split(rng_key, 2)

        # if env_obs is None, then this is the first time calling train and we prepare the environment
        if not self.state.initialized:
            if self.jax_env:
                env_obs, env_states, _ = self.env_reset(reset_rng_key)
            else:
                env_obs, _ = self.env.reset()
                env_states = None

            ep_lens, ep_rets, dones = (
                np.zeros(self.cfg.num_envs),
                np.zeros(self.cfg.num_envs),
                np.zeros(self.cfg.num_envs),
            )
            self.state = self.state.replace(
                ep_lens=ep_lens,
                ep_rets=ep_rets,
                dones=dones,
                env_obs=env_obs,
                env_states=env_states,
                rng_key=rng_key,
                initialized=True,
            )

        start_step = self.state.total_env_steps

        if verbose:
            pbar = tqdm(total=steps + self.state.total_env_steps, initial=start_step)

        env_rollout_size = self.cfg.steps_per_env * self.cfg.num_envs

        while self.state.total_env_steps < start_step + steps:
            rng_key, train_rng_key = jax.random.split(self.state.rng_key, 2)
            self.state, train_step_metrics = self.train_step(train_rng_key, self.state)
            self.state = self.state.replace(rng_key=rng_key)

            # evaluate the current trained actor periodically
            if (
                self.eval_loop is not None
                and tools.reached_freq(self.state.total_env_steps, self.cfg.eval_freq, step_size=env_rollout_size)
                and self.state.total_env_steps > self.cfg.num_seed_steps
            ):
                rng_key, eval_rng_key = jax.random.split(rng_key, 2)
                eval_results = self.evaluate(
                    eval_rng_key,
                    num_envs=self.cfg.num_eval_envs,
                    steps_per_env=self.cfg.eval_steps,
                    eval_loop=self.eval_loop,
                    params=self.state.ac.actor,
                    apply_fn=self.state.ac.act,
                )
                self.logger.store(
                    tag="test",
                    ep_ret=eval_results["eval_ep_rets"],
                    ep_len=eval_results["eval_ep_lens"],
                )
                self.logger.store(tag="test_stats", **eval_results["stats"])
                self.logger.log(self.state.total_env_steps)
                self.logger.reset()

            # log training metrics

            if verbose:
                pbar.update(n=env_rollout_size)
            self.logger.store(tag="train", **train_step_metrics.train)
            self.logger.store(tag="train_stats", **train_step_metrics.train_stats)
            # log time information
            total_time = time.time() - train_start_time
            if tools.reached_freq(self.state.total_env_steps, self.cfg.log_freq):
                self.logger.store(tag="time", **train_step_metrics.time)
                self.logger.store(
                    tag="time",
                    total=total_time,
                    SPS=self.state.total_env_steps / total_time,
                    total_env_steps=self.state.total_env_steps,
                )

            # log and export the metrics
            self.logger.log(self.state.total_env_steps)
            self.logger.reset()

            # save checkpoints. Note that the logger auto saves upon metric improvements
            if tools.reached_freq(self.state.total_env_steps, self.cfg.save_freq):
                self.save(os.path.join(self.logger.model_path, f"ckpt_{self.state.total_env_steps}.jx"))

    def train_step(self, rng_key: PRNGKey, state: SACTrainState) -> Tuple[SACTrainState, TrainStepMetrics]:
        """
        Perform a single training step

        In SAC this is composed of collecting cfg.steps_per_env * cfg.num_envs of interaction data with a random sample or policy (depending on cfg.num_seed_steps)
        then performing gradient updates

        TODO: If a jax-env is used, this step is jitted
        """

        ac = state.ac

        env_obs = state.env_obs
        env_states = state.env_states
        ep_lens = state.ep_lens
        ep_rets = state.ep_rets
        total_env_steps = state.total_env_steps
        training_steps = state.training_steps

        train_custom_stats = defaultdict(list)
        train_metrics = dict()
        train_metrics["ep_ret"] = []
        train_metrics["ep_len"] = []
        time_metrics = dict()

        # perform a rollout
        # TODO make this buffer collection jittable
        rollout_time_start = time.time()
        for _ in range(self.cfg.steps_per_env):
            rng_key, env_rng_key = jax.random.split(rng_key, 2)
            (actions, next_env_obs, next_env_states, rewards, terminations, truncations, infos,) = self._env_step(
                env_rng_key,
                env_obs,
                env_states,
                ac.actor,
                seed=total_env_steps <= self.cfg.num_seed_steps,
            )

            dones = terminations | truncations
            dones = np.array(dones)
            rewards = np.array(rewards)
            # TODO: handle dict observations + is there a more memory efficient way to store o_{t} and o_{t+1} without repeating a lot?
            true_next_env_obs = next_env_obs.copy()
            ep_lens += 1
            ep_rets += rewards

            masks = ((~dones) | (truncations)).astype(float)
            if dones.any():
                # note for continuous task wrapped envs where there is no early done, all envs finish at the same time unless
                # they are staggered. So masks is never false.
                # if you want to always value bootstrap set masks to true.
                for i, d in enumerate(dones):
                    if d:
                        train_metrics["ep_ret"].append(ep_rets[i])
                        train_metrics["ep_len"].append(ep_lens[i])
                        true_next_env_obs[i] = infos["final_observation"][i]
                        final_info = infos["final_info"][i]
                        if "stats" in final_info:
                            for k in final_info["stats"]:
                                train_custom_stats[k].append(final_info["stats"][k])
                ep_lens[dones] = 0.0
                ep_rets[dones] = 0.0

            self.replay_buffer.store(
                env_obs=env_obs,
                reward=rewards,
                action=actions,
                mask=masks,
                next_env_obs=true_next_env_obs,
            )

            env_obs = next_env_obs
            env_states = next_env_states

        rollout_time = time.time() - rollout_time_start
        time_metrics["rollout_time"] = rollout_time

        # update policy
        if self.state.total_env_steps >= self.cfg.num_seed_steps:
            update_time_start = time.time()
            update_actor = training_steps % self.cfg.actor_update_freq == 0
            update_target = training_steps % self.cfg.target_update_freq == 0
            for _ in range(self.cfg.grad_updates_per_step):
                rng_key, update_rng_key, sample_key = jax.random.split(rng_key, 3)
                batch = self.replay_buffer.sample_random_batch(sample_key, self.cfg.batch_size)
                batch = TimeStep(**batch)
                ac, aux = self.update_parameters(
                    update_rng_key,
                    ac,
                    batch,
                    update_actor,
                    update_target,
                )
            update_time = time.time() - update_time_start
            critic_update_aux: loss.CriticUpdateAux = aux["critic_update_aux"]
            actor_update_aux: loss.ActorUpdateAux = aux["actor_update_aux"]
            temp_update_aux: loss.TempUpdateAux = aux["temp_update_aux"]

            train_metrics["critic_loss"] = float(critic_update_aux.critic_loss)
            train_metrics["q1"] = float(critic_update_aux.q1)
            train_metrics["q2"] = float(critic_update_aux.q2)
            train_metrics["temp"] = float(temp_update_aux.temp)
            if update_actor:
                train_metrics["actor_loss"] = float(actor_update_aux.actor_loss)
                train_metrics["entropy"] = float(actor_update_aux.entropy)
                train_metrics["target_entropy"] = float(self.cfg.target_entropy)
                if self.cfg.learnable_temp:
                    train_metrics["temp_loss"] = float(temp_update_aux.temp_loss)
            time_metrics["update_time"] = update_time

        state = state.replace(
            ac=ac,
            ep_lens=ep_lens,
            ep_rets=ep_rets,
            env_obs=env_obs,
            env_states=env_states,
            total_env_steps=total_env_steps + self.cfg.num_envs * self.cfg.steps_per_env,
            training_steps=training_steps + 1,
            dones=dones,
        )

        return state, TrainStepMetrics(time=time_metrics, train=train_metrics, train_stats=train_custom_stats)

    @partial(jax.jit, static_argnames=["self", "update_actor", "update_target"])
    def update_parameters(
        self,
        rng_key: PRNGKey,
        ac: ActorCritic,
        batch: TimeStep,
        update_actor: bool,
        update_target: bool,
    ) -> Tuple[ActorCritic, Any]:
        """
        Update actor critic parameters using the given batch
        """
        rng_key, critic_update_rng_key = jax.random.split(rng_key, 2)
        new_critic, critic_update_aux = loss.update_critic(
            critic_update_rng_key,
            ac,
            batch,
            self.cfg.discount,
            self.cfg.backup_entropy,
        )
        # init dummy values
        new_actor, actor_update_aux = ac.actor, loss.ActorUpdateAux()
        new_temp, temp_update_aux = ac.temp, loss.TempUpdateAux(temp=ac.temp())
        new_target = ac.target_critic

        if update_target:
            new_target = loss.update_target(ac.critic, ac.target_critic, self.cfg.tau)
        if update_actor:
            rng_key, actor_update_rng_key = jax.random.split(rng_key, 2)
            new_actor, actor_update_aux = loss.update_actor(actor_update_rng_key, ac, batch)
            if self.cfg.learnable_temp:
                new_temp, temp_update_aux = loss.update_temp(ac.temp, actor_update_aux.entropy, self.cfg.target_entropy)
        ac = ac.replace(actor=new_actor, critic=new_critic, target_critic=new_target, temp=new_temp)
        return (
            ac,
            dict(
                critic_update_aux=critic_update_aux,
                actor_update_aux=actor_update_aux,
                temp_update_aux=temp_update_aux,
            ),
        )

    def state_dict(self):
        # TODO add option to save buffer?
        ac = flax.serialization.to_bytes(self.state.ac)
        state_dict = dict(
            train_state=self.state.replace(ac=ac),
            logger=self.logger.state_dict(),
        )
        if self.cfg.save_buffer_in_checkpoints:
            state_dict["replay_buffer"] = self.replay_buffer
        return state_dict

    def save(self, save_path: str):

        stime = time.time()
        state_dict = self.state_dict()
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            # TODO replace pickle with something more efficient for replay buffers?
        print(f"Saving Checkpoint {save_path}.", "Time:", time.time() - stime)

    def load_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        print(f"Loading Checkpoint {load_path}", state_dict["logger"])
        self.load(state_dict)

    def load(self, data):
        ac = flax.serialization.from_bytes(self.state.ac, data["train_state"].ac)
        # use serialized ac model
        self.state: SACTrainState = data["train_state"].replace(ac=ac)
        # set initialized to False so previous env data is reset if it's not a jax env with env states we can start from
        if not self.jax_env:
            self.state = self.state.replace(initialized=False)
        self.logger.load(data["logger"])
        if "replay_buffer" in data:
            replay_buffer: GenericBuffer = data["replay_buffer"]
            print(f"Loading replay buffer which contains {replay_buffer.size() * replay_buffer.num_envs} interactions")
            self.replay_buffer = replay_buffer
