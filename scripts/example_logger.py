import gym
import torch

from paper_rl.modelfree.ppo import PPO
from paper_rl.architecture.ac.mlp import MLPActorCritic
from paper_rl.logger import Logger
logger = Logger()
logger.store(tag="train", append=True, reward=-1)
logger.store(tag="train", append=True, reward=-0.5)
logger.store(tag="test", append=False, interactions=200)
logger.store(tag="train", append=False, interactions=200)
stats = logger.log(step=0)
logger.pretty_print_table(stats)