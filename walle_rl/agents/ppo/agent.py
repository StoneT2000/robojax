from typing import Dict, Optional

import distrax
from walle_rl.agents.ppo.buffer import PPOBuffer, Batch
from walle_rl.architecture.ac.core import ActorCritic
from walle_rl.agents.base import Policy
from walle_rl.buffer.buffer import GenericBuffer
from walle_rl.optim.pg import clipped_surrogate_pg_loss
import jax.numpy as jnp
class PPO(Policy):
    ac: ActorCritic
    buffer: PPOBuffer
    num_envs: int
    gamma: Optional[float] = 0.99
    gae_lambda: Optional[float] = 0.97
    clip_ratio: Optional[float] = 0.2
    ent_coef: Optional[float] = 0.0
    pi_coef: Optional[float] = 1.0
    vf_coef: Optional[float] = 1.0
    dapg_lambda: Optional[float] = 0.1
    dapg_damping: Optional[float] = 0.99
    target_kl: Optional[float] = 0.01

    def gradient(self):
        batch = self.buffer.sample_batch
        info_a = self.update_actor_parameters(self, )

    def get_actor_loss(self, ac: ActorCritic, batch: Batch):
        obs, act, adv, logp_old = batch["obs_buf"], batch["act_buf"], batch["adv_buf"], batch["logp_buf"]
        
        # ac.pi.´val()
        pi, logp = ac.pi(obs=obs, act=act)
        # ac.pi.train()
        prob_ratios = jnp.exp(logp - logp_old)
        loss_pi = clipped_surrogate_pg_loss(prob_ratios_t=prob_ratios, adv_t=adv, clip_ratio=self.clip_ratio)

        entropy = -logp.mean()

        info = dict(
            loss_pi=loss_pi,
            entropy=entropy
        )
        return info
    def get_critic_loss(self, ac: ActorCritic, batch: Batch):
        obs, ret = batch["obs_buf"], batch["ret_buf"]
        loss_v = ((ac.v(obs) - ret) ** 2).mean()

        return dict(
            loss_v=loss_v
        )