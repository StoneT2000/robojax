# Robojax

A high-performance reinforcement learning library in jax specialized for robotic learning. Works out of the box with jax based environments like [Gymnax](https://github.com/RobertTLange/gymnax) and [Brax](https://github.com/google/brax/tree/main/brax) and non-jax based environments like ...

<!-- <img src="https://user-images.githubusercontent.com/35373228/160072285-fb65294b-f6a6-4028-b60a-ac774191ac85.jpg" width=200/> -->

How is it specialized? It includes popular algorithms often used in robotic learning research like PPO and SAC, as well as architectures often used for visual / 3D RL like transformers, point nets, etc.

## Setup

I highly recommend using conda. Otherwise you can try and install all the packages yourself (at your own risk of not getting reproducible results)

```
conda env create -f environment.yml
```