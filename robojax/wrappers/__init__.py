"""
Env Wrappers for all envs tested in robojax.

Generally all non-jax based environments have wrappers applied such that they map to a Gymnasium API environment or a Gymnasium VectorEnv API for those that are already parallelized
For new environments as long as they follow Gymnasium then it should work in this library

For all jax based environments, wrappers are applied such that they map to a modified Gymnax API that conforms to the Gymnasium API with state.

Importantly, for recording videos a wrapper is applied to make a Gymnax style env into a Gymnasium style VectorEnv and a RecordVideo wrapper is applied on top.

"""
