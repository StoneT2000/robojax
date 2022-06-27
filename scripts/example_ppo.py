import walle_rl.agents.ppo

# define model here, either your self or using our archs
model = Model()

# call training in a functional way, like jax, but also supportive of pytorch



# Compose modules
"""
We need a model, e.g. MLP that does predicts actions

Then we need a exploration head


We need an algo, PPO, SAC, ... which can be deconstructed into
- optimization (DPG, clipped surrogate pg loss...)

We need an exploration strategy (sampler, intrinsic rewards) (which might be part of exploration head)

We need a environment

We need a experiment loop 
( which is own pipeline. E.g. rollout, learn, )

Run!

"""