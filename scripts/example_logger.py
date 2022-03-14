import gym
import torch

from paper_rl.modelfree.ppo import PPO
from paper_rl.architecture.ac.mlp import MLPActorCritic
from paper_rl.logger import Logger
from omegaconf import OmegaConf
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("fpath", type=str)

# parser.add_argument("--train_from_scratch", action="store_true")
# exp_params

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cpu", type=int, default=4)
args = parser.parse_args()
listkwargs = [f"{a}={b}" for a,b in args._get_kwargs()]
conf = OmegaConf.from_cli(listkwargs)
print(conf)
logger = Logger(wandb=True, workspace="examplelog", exp_name="", cfg=conf)
logger.store(tag="train", append=True, reward=-1)
logger.store(tag="train", append=True, reward=-0.5)
logger.store(tag="test", append=False, interactions=200)
logger.store(tag="train", append=False, interactions=200)
stats = logger.log(step=0)
logger.pretty_print_table(stats)