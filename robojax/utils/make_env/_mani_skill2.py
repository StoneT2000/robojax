import gymnasium as gym
from gymnasium.wrappers import RecordVideo


try:
    import mani_skill2.envs  # NOQA

    import robojax.wrappers._mani_skill2 as ms2wrappers
except ImportError:
    pass


def is_mani_skill2_env(env_id: str):
    try:
        import mani_skill2.envs  # NOQA
    except ImportError:
        return False
    from mani_skill2.utils.registration import REGISTERED_ENVS

    return env_id in REGISTERED_ENVS


def env_factory(env_id: str, idx: int, seed: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[]):
    internal_wrappers = []
    internal_wrappers.append(lambda x: ms2wrappers.ManiSkill2Wrapper(x))
    internal_wrappers.append(lambda x: ms2wrappers.ContinuousTaskWrapper(x))

    def _init():
        env = gym.make(env_id, disable_env_checker=True, **env_kwargs)
        for wrapper in internal_wrappers:
            env = wrapper(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordVideo(env, record_video_path)
        return env

    return _init