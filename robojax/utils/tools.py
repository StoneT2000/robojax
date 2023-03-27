import numpy as np
from chex import Array


def any_to_numpy(x: Array):
    return np.array(x)


def copy_arr(x: Array):
    return x.copy()


def reached_freq(t, freq, step_size=1):
    """
    Returns False if `freq > 0` andd time `t` has reached the frequency. Gives a leeway of size `step_size - 1`.

    `step_size=1` is equivalent to checking if `t % freq == 0`.
    """
    if freq > 0 and (t - step_size) // freq < t // freq:
        return True
    return False
