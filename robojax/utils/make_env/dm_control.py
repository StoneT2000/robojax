try:
    pass
except:
    pass


def is_dm_control_env(env_id: str):
    try:
        from dm_control import suite
    except:
        return False
    suite.BENCHMARKING
