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
| `total_timesteps` | `train.epochs * ppo.steps_per_epoch * ppo.num_envs` | Total number of interactions |

Some other notes: CleanRL reports episode data (return, length) the moment the episode ends in PPO. RoboJax reports aggregated metrics (min, max, mean) of the same data for the same rollout when reset_env = True, otherwise reports the saame way

CleanRL also does not explicitly reset all the environments to `t=0` between rollouts (saame behavior achieved with ppo.reset_env = False)

CleanRL global_step is the number of interactions sampled. RoboJax logs results based on global_step as well.


```
python scripts/experiment_ppo.py env.env_id="CartPole-v1" env.max_episode_steps=500 eval_env.max_episode_steps=500
```


### Comparing with brax ppo

| Brax PPO      | RoboJax | Description |
| ----------- | ----------- | -------- |
| ` env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat` (computed) | `ppo.steps_per_env * ppo.num_envs` | Total number of interaction steps before policy update |
| `num_envs` | `ppo.num_envs` | Number of parallel environments |
| `batch_size * unroll_length`| `batch_size` | Size of sampled batch of data from replay buffer during policy updaates |
| `num_minibatches * num_updates_per_batch` | `grad_updates_per_step` | number of gradient steps

collects data in shape (batch_size * num_minibatches, unroll_length)

during sgd step (which has num_minibatches steps), this is reshaped to (num_minibatches, -1) + unroll_length? then scanover num_minibatches and mini batch step of size 2048 each.















```
python scripts/experiment_sac.py scripts/cfgs/sac/hopper_mujoco.yml logger.exp_name="mujoco/Hopper-v4_sac_s0" logger.wandb=True seed=0 logger.clear_out=True

python scripts/experiment_sac.py scripts/cfgs/sac/hopper_brax.yml logger.exp_name="mujoco/hopper-brax_sac_s0" logger.wandb=True seed=0 logger.clear_out=True
```