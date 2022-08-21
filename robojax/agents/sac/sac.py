from robojax.agents.base import BasePolicy
from robojax.agents.sac.config import SACConfig


class SAC(BasePolicy):
    def __init__(
        self,
        jax_env: bool,
        env=None,
        env_step=None,
        env_reset=None,
        cfg: SACConfig = {},
    ):
        super().__init__(jax_env, env, env_step, env_reset)
        