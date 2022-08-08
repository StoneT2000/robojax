import distrax
import scipy.signal
from chex import Array


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    # TODO replace with jax
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def _log_prob_from_distribution(dist: distrax.Distribution, x: Array):
    return dist.log_prob(x)
