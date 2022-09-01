# Robojax

A high-performance reinforcement learning library in jax specialized for robotic learning. Works out of the box with jax based environments/engines like [Gymnax](https://github.com/RobertTLange/gymnax) and [Brax](https://github.com/google/brax/tree/main/brax), as well as non-jax based environments/engines like [Mujoco](https://github.com/deepmind/mujoco), [SAPIEN](https://github.com/haosulab/SAPIEN), [MetaWorld](https://github.com/rlworkgroup/metaworld) etc. 

<!-- <img src="https://user-images.githubusercontent.com/35373228/160072285-fb65294b-f6a6-4028-b60a-ac774191ac85.jpg" width=200/> -->

How is it specialized? It includes popular algorithms often used in robotic learning research like PPO and SAC, as well as supporting architectures and workflows often used for visual / 3D RL like transformers, point nets, etc.

And some more recent advances may also be included

## Setup

It's highly recommended to use conda. Otherwise you can try and install all the packages yourself (at your own risk of not getting reproducible results)

```
conda env create -f environment.yml
```

To install jax with cuda support, follow the instructions on their [README](https://github.com/google/jax).

## Usage

## Benchmarking
See https://wandb.ai/stonet2000/robojax?workspace=user-stonet2000 for all benchmarked results on this library

To benchmark the code yourself, see https://github.com/StoneT2000/robojax/tree/main/scripts/baselines.md