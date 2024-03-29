import gymnasium as gym


try:
    import mani_skill2.envs  # NOQA

    import robojax.wrappers._mani_skill2 as ms2wrappers

    # from mani_skill2.utils.wrappers import RecordEpisode
    from robojax.wrappers._mani_skill2_record_gymnasium import RecordEpisode
except ImportError:
    pass


def is_mani_skill2_env(env_id: str):
    try:
        import mani_skill2.envs  # NOQA
    except ImportError:
        return False
    from mani_skill2.utils.registration import REGISTERED_ENVS

    return env_id in REGISTERED_ENVS or env_id in ["PickCube-v1", "PegInsertionSide-v1", "StackCube-v1"]


def env_factory(env_id: str, idx: int, seed: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[]):
    internal_wrappers = []
    render_mode = "rgb_array"
    if "render_mode" in env_kwargs:
        render_mode = env_kwargs["render_mode"]
    internal_wrappers.append(lambda x: ms2wrappers.ManiSkill2Wrapper(x, render_mode=render_mode))
    if "continuous_task" not in env_kwargs or env_kwargs["continuous_task"]:
        internal_wrappers.append(lambda x: ms2wrappers.ContinuousTaskWrapper(x))

    def _init():
        env = gym.make(env_id, disable_env_checker=True, **env_kwargs)
        for wrapper in internal_wrappers:
            env = wrapper(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordEpisode(env, record_video_path, save_trajectory=False, info_on_video=True)
        return env

    return _init
