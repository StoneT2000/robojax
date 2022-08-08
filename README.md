# Robojax

A high-performance reinforcement learning library in jax specialized for robotic learning

<img src="https://user-images.githubusercontent.com/35373228/160072285-fb65294b-f6a6-4028-b60a-ac774191ac85.jpg" width=200/>

My personal RL library (at the moment). Frequently changed and edited, but feel free to use as potential reference code (its mostly correct)

Implements PPO, DAPG w/ PPO, and GAIL (probably broken)

Main Features:
- Always seeded, we love reproducibility
- Highly configurable with functional capabilities. You can pull out specific parts of algorithms out and tweak it (e.g. the PPO update is written functionally)
- WALL-E might have endorsed this repository

## Setup

I highly recommend using conda. Otherwise you can try and install all the packages yourself (at your own risk of not getting reproducible results)

```
conda env create -f environment.yml
```

## Contributors



- name credits to @jonzamora
