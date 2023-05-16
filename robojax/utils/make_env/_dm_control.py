from gymnasium.wrappers import RecordVideo


try:
    from robojax.wrappers._dm_control import DMContolEnv
except ImportError:
    pass


def is_dm_control_env(env_id: str):
    try:
        from dm_control import suite
    except ImportError:
        pass
    domain_name, task_name = env_id.split("-")
    return (domain_name, task_name) in suite.BENCHMARKING


def env_factory(env_id, idx, seed, env_kwargs=dict(), record_video_path: str = None, wrappers=[]):
    def _init():
        domain_name, task_name = env_id.split("-")
        if "task_kwargs" not in env_kwargs:
            env_kwargs["task_kwargs"] = {}
        env_kwargs["task_kwargs"]["random"] = seed
        env = DMContolEnv(domain_name=domain_name, task_name=task_name, **env_kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True)
        return env

    return _init
