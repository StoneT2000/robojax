import numpy as np
from chex import Array


def any_to_numpy(x: Array):
    return np.array(x)


def copy_arr(x: Array):
    return x.copy()
