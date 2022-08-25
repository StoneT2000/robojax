from typing import Any, Callable, Tuple

from chex import PRNGKey

from robojax.data.loop import EnvAction, EnvObs, EnvState, GymLoop, JaxLoop
from robojax.utils.spaces import get_action_dim, get_obs_shape


class BasePolicy:
    def __init__(
        self,
        jax_env: bool,
        env=None,
    ) -> None:
        """
        Base class for a policy

        Equips it with loopers
        """
        assert env is not None
        self.jax_env = jax_env
        if jax_env:
            
            self.env_step: Callable[
                [PRNGKey, EnvState, EnvAction],
                Tuple[EnvObs, EnvState, float, bool, Any],
            ] = env.step
            self.env_reset: Callable[[PRNGKey], Tuple[EnvObs, EnvState]] = env.reset
            import gymnax.environments.environment
            self.env: gymnax.environments.environment.Environment = env

            self.loop = JaxLoop(env_reset=self.env.reset, env_step=self.env.step)
            self.observation_space = self.env.observation_space()
            self.action_space = self.env.action_space()
        else:
            import gym
            self.env: gym.Env = env

            self.loop = GymLoop(self.env)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        self.obs_shape = get_obs_shape(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)