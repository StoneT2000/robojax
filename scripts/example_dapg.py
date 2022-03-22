import os.path as osp
import pickle
import gym
import numpy as np
import torch
from paper_rl.cfg import parse_cfg
from paper_rl.common.rollout import Rollout
from paper_rl.logger.logger import Logger

from paper_rl.modelfree.ppo import PPO
from paper_rl.architecture.ac.mlp import MLPActorCritic
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.env_util import make_vec_env

# 2 ways to work with envs
# mpi it and have one env per process, each process has its own copy
# or use stable baselines vecenv style. - which also makes it easy to utilize the GPU.

if __name__ == "__main__":
    # env_id = "Pendulum-v0"
    cfg = parse_cfg(default_cfg_path=osp.join(osp.dirname(__file__), "sample_configs/default_dapg.yml"))
    env_id = cfg["env_id"]
    num_cpu = cfg["cpu"]
    seed = cfg["seed"]
    env = make_vec_env(env_id, num_cpu, seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(64, 64))
    pi_optimizer = torch.optim.Adam(model.pi.parameters(), lr=1e-4)
    vf_optimizer = torch.optim.Adam(model.v.parameters(), lr=4e-4)
    
    logger = Logger(tensorboard=False, wandb=False, cfg=cfg, workspace=cfg["workspace"], exp_name=cfg["exp_name"])
    steps_per_epoch = cfg["steps_per_epoch"] // num_cpu
    batch_size = cfg["batch_size"]
    algo = PPO(
        ac=model,
        env=env,
        num_envs=num_cpu,
        action_space=env.action_space,
        observation_space=env.observation_space,
        logger=logger,
        steps_per_epoch=steps_per_epoch,
        gamma=cfg["gamma"],
        train_iters=cfg["train_iters"],  # 80 // (steps_per_epoch * num_cpu // batch_size)
        dapg_damping=cfg["dapg_damping"],
        dapg_lambda=cfg["dapg_lambda"],
        pi_coef=cfg["pi_coef"]
    )

    def train_callback(epoch):
        stats = logger.log(step=epoch)
        filtered = {}
        for k in stats.keys():
            if (
                "Epoch" in k
                or "TotalEnvInteractions" in k
                or "EpRet" in k
                or "EpLen" in k
                # or "VVals" in k
                or "LossPi_avg" in k
                or "LossDAPGActor_avg" in k
                or "PPOLoss_avg" in k
                or "dapg_lambda" in k
                or "UpdateTime" in k
                or "RolloutTime" in k
            ):
                filtered[k] = stats[k]
        logger.pretty_print_table(filtered)
        logger.reset()

    # sample demo trajectories
    with open("./scripts/expert_cartpole.pkl", "rb") as f:
        expert_trajectories = pickle.load(f)

    demo_trajectories = dict(observations=[], actions=[])
    for traj in expert_trajectories:
        demo_trajectories["observations"].append(traj["observations"][:-1])
        demo_trajectories["actions"].append(traj["actions"])
    demo_trajectories["observations"] = torch.as_tensor(np.vstack(demo_trajectories["observations"]), dtype=torch.float32)
    demo_trajectories["actions"] = torch.as_tensor(np.vstack(demo_trajectories["actions"]), dtype=torch.float32)
    def demo_trajectory_sampler(batch_size):
        transition_inds = np.random.randint(0, len(demo_trajectories["observations"]), size=batch_size)
        return dict(
            observations=demo_trajectories["observations"][transition_inds],
            actions=demo_trajectories["actions"][transition_inds]
        )
    algo.train(
        max_ep_len=1000,
        start_epoch=0,
        n_epochs=cfg["epochs"],
        pi_optimizer=pi_optimizer,
        vf_optimizer=vf_optimizer,
        batch_size=batch_size,
        rollout_callback=None,
        train_callback=train_callback,
        demo_trajectory_sampler=demo_trajectory_sampler,
        dapg_nll_loss=cfg["dapg_nll_loss"]
    )
    env.close()

    eval_env = make_vec_env(env_id, 1, seed=seed)
    obs = eval_env.reset()

    for i in range(1000):
        with torch.no_grad():
            action = model.act(torch.tensor(obs), deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
        if done.any():
            print(info)
            break
    eval_env.close()

    eval_env = make_vec_env(env_id, 2, seed=seed)
    obs = eval_env.reset()
    
    rollout = Rollout()
    def policy(o):
        o = torch.as_tensor(o, dtype=torch.float32)
        return model.act(o, deterministic=True)
    expert_trajectories = rollout.collect_trajectories(policy, eval_env, n_trajectories=10, n_envs=2,)
    eval_env.close()
    for e in expert_trajectories:
        logger.store(tag="test", append=True, EpLen=len(e["observations"] - 1))

    stats = logger.log(cfg["epochs"] - 1)
    logger.pretty_print_table(stats)

    logger.close()