from typing import Any, Callable, Tuple

from chex import PRNGKey

from robojax.data.loop import EnvAction, EnvObs, EnvState


class BasePolicy:
    def __init__(
        self,
        jax_env: bool,
        env=None,
        env_step=None,
        env_reset=None,
    ) -> None:
        self.jax_env = jax_env
        if jax_env:
            assert env is None
            assert env_step is not None
            assert env_reset is not None
            self.env_step: Callable[
                [PRNGKey, EnvState, EnvAction],
                Tuple[EnvObs, EnvState, float, bool, Any],
            ] = env_step
            self.env_reset: Callable[[PRNGKey], Tuple[EnvObs, EnvState]] = env_reset
        else:
            assert env is not None
            assert env_step is None
            assert env_reset is None
            import gym

            self.env: gym.Env = env
