import collections
from typing import Iterable, Iterator, Tuple, Union

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jax import config as jax_config

PRNGSequenceState = Tuple[PRNGKey, Iterable[PRNGKey]]
DEFAULT_PRNG_RESERVE_SIZE = 1

"""PRNGSequence magic from dm-haiku """


def assert_is_prng_key(key: PRNGKey):
    """Asserts that the given input looks like a `jax.random.PRNGKey`."""
    type_error = ValueError(
        "The provided key is not a JAX PRNGKey but a " f"{type(key)}:\n{key}"
    )
    if (
        hasattr(jax.config, "jax_enable_custom_prng")
        and jax.config.jax_enable_custom_prng
    ):
        if not isinstance(key, jax.random.KeyArray):
            raise type_error
    if not hasattr(key, "shape"):
        raise type_error
    if hasattr(key, "dtype"):
        config_hint = ""
        if hasattr(jax.random, "default_prng_impl"):
            default_impl = jax.random.default_prng_impl()
            expected_shapes = (default_impl.key_shape,)
            if default_impl.key_shape != (2,):
                # Default PRNG impl is set to something different from threefry.
                config_hint = (
                    "\nHint: jax_default_prng_impl has been set to "
                    f"'{jax_config.jax_default_prng_impl}', the shape "
                    "mismatch might be because a jax.random.PRNGKey was "
                    "created before this flag was set."
                )
        else:
            # Check for the shapes of the known PRNG impls (threefry and RBG)
            expected_shapes = ((2,), (4,))
        if key.shape not in expected_shapes or key.dtype != jnp.uint32:
            expected_shapes_str = " or ".join(map(str, expected_shapes))
            raise ValueError(
                "Provided key did not have expected shape and/or dtype: "
                f"expected=(shape={expected_shapes_str}, dtype=uint32), "
                f"actual=(shape={key.shape}, dtype={key.dtype}){config_hint}"
            )


class PRNGSequence(Iterator[PRNGKey]):
    """Iterator of JAX random keys.
    >>> seq = hk.PRNGSequence(42)  # OR pass a jax.random.PRNGKey
    >>> key1 = next(seq)
    >>> key2 = next(seq)
    >>> assert key1 is not key2
    If you know how many keys you will want then you can use :meth:`reserve` to
    more efficiently split the keys you need::
    >>> seq.reserve(4)
    >>> keys = [next(seq) for _ in range(4)]
    """

    __slots__ = ("_key", "_subkeys")

    def __init__(self, key_or_seed: Union[PRNGKey, int, PRNGSequenceState]):
        """Creates a new :class:`PRNGSequence`."""
        if isinstance(key_or_seed, tuple):
            key, subkeys = key_or_seed
            assert_is_prng_key(key)
            for subkey in subkeys:
                assert_is_prng_key(subkey)
            self._key = key
            self._subkeys = collections.deque(subkeys)
        else:
            if isinstance(key_or_seed, int):
                key_or_seed = jax.random.PRNGKey(key_or_seed)
            # A seed value may also be passed as an int32-typed scalar ndarray.
            elif (
                hasattr(key_or_seed, "shape")
                and (not key_or_seed.shape)
                and hasattr(key_or_seed, "dtype")
                and key_or_seed.dtype == jnp.int32
            ):
                key_or_seed = jax.random.PRNGKey(key_or_seed)
            else:
                assert_is_prng_key(key_or_seed)
            self._key = key_or_seed
            self._subkeys = collections.deque()

    def reserve(self, num):
        """Splits additional ``num`` keys for later use."""
        if num > 0:
            # When storing keys we adopt a pattern of key0 being reserved for future
            # splitting and all other keys being provided to the user in linear order.
            # In terms of jax.random.split this looks like:
            #
            #     key, subkey1, subkey2 = jax.random.split(key, 3)  # reserve(2)
            #     key, subkey3, subkey4 = jax.random.split(key, 3)  # reserve(2)
            #
            # Where subkey1->subkey4 are provided to the user in order when requested.
            new_keys = tuple(jax.random.split(self._key, num + 1))
            self._key = new_keys[0]
            self._subkeys.extend(new_keys[1:])

    def reserve_up_to_full(self):
        num = DEFAULT_PRNG_RESERVE_SIZE - len(self._subkeys)
        if num > 0:
            self.reserve(num)
        else:
            sliced_subkeys = list(self._subkeys)[:DEFAULT_PRNG_RESERVE_SIZE]
            self._subkeys = collections.deque(sliced_subkeys)

    @property
    def internal_state(self) -> PRNGSequenceState:
        return self._key, tuple(self._subkeys)

    def replace_internal_state(self, state: PRNGSequenceState):
        key, subkeys = state
        assert_is_prng_key(key)
        for subkey in subkeys:
            assert_is_prng_key(subkey)
        self._key = key
        self._subkeys = collections.deque(subkeys)

    def __next__(self) -> PRNGKey:
        if not self._subkeys:
            self.reserve(DEFAULT_PRNG_RESERVE_SIZE)
        return self._subkeys.popleft()

    next = __next__

    def take(self, num) -> Tuple[PRNGKey, ...]:
        self.reserve(max(num - len(self._subkeys), 0))
        return tuple(next(self) for _ in range(num))
