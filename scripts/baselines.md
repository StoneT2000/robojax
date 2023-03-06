# Baselines

Running all baselines

To log to weights and biases, add `logging.wandb=True`.

```
python scripts/experiment_sac.py env_id=Hopper-v3
python scripts/experiment_ppo.py env_id=Ant-v3
```

Matching OpenRL Benchmark's PPO.

Translating CleanRL hyperparameters to Robojax hyperparameters for PPO

| CleanRL      | RoboJax | Description |
| ----------- | ----------- | -------- |
| `batch_size` (computed) | `ppo.steps_per_epoch * ppo.num_envs` | Total number of interaction steps before policy update |
| `num_envs`   | `ppo.num_envs` | Number of parallel environments |
| `num_steps` | `ppo.steps_per_epoch` | Number of steps to run per parallel environment |
| `minibatch_size`| `batch_size` | Size of sampled batch of data from replay buffer during policy updaates |
| `update_epochs` | `ppo.update_iters / (ppo.steps_per_epoch * ppo.num_envs // batch_size)` (computed) | Number of iterations over the entire replay buffer, with each iteration consisting of a number of gradient updates. In Robojax you directly control the number of updates. |
| `update_epochs * (batch_size // mini_batch_size)` (computed) | `ppo.update_iters` | Number of gradient updates

Some other notes: CleanRL reports episode data (return, length) the moment the episode ends in PPO. RoboJax reports aggregated metrics (min, max, mean) of the same data for the same rollout.

CleanRL also does not explicitly reset all the environments to `t=0` between rollouts. 

```
python scripts/experiment_ppo.py env.env_id="CartPole-v1" env.max_episode_steps=500 eval_env.max_episode_steps=500
```