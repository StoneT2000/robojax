try:
    import mani_skill2.envs  # NOQA
except:
    pass


def is_mani_skill2_env(env_id: str):
    try:
        import mani_skill2.envs  # NOQA
    except:
        return False
    from mani_skill2.utils.registration import REGISTERED_ENVS

    return env_id in REGISTERED_ENVS
