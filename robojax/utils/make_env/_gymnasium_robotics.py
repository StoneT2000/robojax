import gymnasium
from gymnasium.wrappers import RecordVideo


try:
    from robojax.wrappers._gymnasium_robotics import ContinuousTaskWrapper
except ImportError:
    pass


def is_gymnasium_robotics_env(env_id: str):
    try:
        pass
    except ImportError:
        return False
    if env_id not in gymnasium.registry:
        return False
    return "gymnasium_robotics" in gymnasium.registry[env_id].entry_point


def env_factory(env_id, idx, seed, record_video_path, env_kwargs, wrappers=[]):
    def _init():
        env = gymnasium.make(env_id, disable_env_checker=True, **env_kwargs)
        env = ContinuousTaskWrapper(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True)
        return env

    return _init
