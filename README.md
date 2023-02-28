# Robojax

A high-performance reinforcement learning library in jax specialized for robotic learning. Works out of the box with jax based environments/engines like [Gymnax](https://github.com/RobertTLange/gymnax) and [Brax](https://github.com/google/brax/tree/main/brax), as well as non-jax based environments/engines like [Mujoco](https://github.com/deepmind/mujoco), [SAPIEN](https://github.com/haosulab/SAPIEN), [MetaWorld](https://github.com/rlworkgroup/metaworld) etc. 

<!-- <img src="https://user-images.githubusercontent.com/35373228/160072285-fb65294b-f6a6-4028-b60a-ac774191ac85.jpg" width=200/> -->

How is it specialized? It includes popular algorithms often used in robotic learning research like PPO and SAC, as well as eventually supporting architectures and workflows often used for visual / 3D RL like transformers, point nets, etc. It further will include more robotics specific approaches such as [Transporter Networks (Zeng et al., 2020)](https://transporternets.github.io/).

## Setup

It's highly recommended to use conda. Otherwise you can try and install all the packages yourself (at your own risk of not getting reproducible results)

```
conda env create -f environment.yml
```

To install jax with cuda support, follow the instructions on their [README](https://github.com/google/jax).

## Organization

The following modules are usually shared between RL algorithms

- JaxLoop / Environment Loop for collecting rollouts
- Evaluation Protocol for evaluating agent during training
- Loggers for logging training and test data
- Generic `Model` interface

Everything else is usually kept inside the RL algorithm module e.g. `robojax.agents.ppo` contains the PPO Config, Actor Critic models, loss functions etc. all separate from e.g. `robojax.agents.sac`.

## Benchmarking
See https://wandb.ai/stonet2000/robojax?workspace=user-stonet2000 for all benchmarked results on this library

To benchmark the code yourself, see https://github.com/StoneT2000/robojax/tree/main/scripts/baselines.md
