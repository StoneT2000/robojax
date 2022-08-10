# Robojax

A high-performance reinforcement learning library in jax specialized for robotic learning. Works out of the box with jax based environments like [Gymnax](https://github.com/RobertTLange/gymnax) and [Brax](https://github.com/google/brax/tree/main/brax) and non-jax based environments like ...

<!-- <img src="https://user-images.githubusercontent.com/35373228/160072285-fb65294b-f6a6-4028-b60a-ac774191ac85.jpg" width=200/> -->

How is it specialized? It includes popular algorithms often used in robotic learning research like PPO and SAC, as well as architectures often used for visual / 3D RL like transformers, point nets, etc.

## Setup

I highly recommend using conda. Otherwise you can try and install all the packages yourself (at your own risk of not getting reproducible results)

```
conda env create -f environment.yml
```

To install jax with cuda support, follow the instructions on their README.

## Benchmarking
See https://wandb.ai/stonet2000/robojax?workspace=user-stonet2000 for all benchmarked results on this library

To benchmark the code yourself, you can run the following:
```
python scripts/experiment.py jax_env=False logger.exp_name="ppo/cartpole_gym"
python scripts/experiment.py jax_env=True logger.exp_name="ppo/cartpole_gymnax"

python scripts/experiment.py jax_env=True logger.exp_name="ppo/ant_brax" logger.wandb=False env_id="ant"
```