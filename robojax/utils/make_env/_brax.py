try:
    from brax import envs

    from robojax.wrappers._brax import BraxGymWrapper
except ImportError:
    pass


def is_brax_env(env_id: str):
    try:
        from brax import envs
    except ImportError:
        return False
    return env_id in envs._envs


def env_factory(env_id, env_kwargs=dict(), record_video_path: str = None, wrappers=[], max_episode_steps=None):
    def _init():
        env = envs.create(env_id, episode_length=None, auto_reset=False, **env_kwargs)
        env = BraxGymWrapper(env, max_episode_steps=max_episode_steps, auto_reset=True)
        for wrapper in wrappers:
            env = wrapper(env)
        # if record_video_path is not None and idx == 0:
        #     env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True) # TODO create a jax env video recording wrapper
        return env

    return _init
