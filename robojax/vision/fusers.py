"""
Code that fuses multiple views (pointcloud or RGBD) into a single view with various projections
"""
from chex import Array
import jax
import jax.numpy as jnp

# def fuse_pointclouds(pcds: Array):

# @jax.jit
# def fused_heightmap(pcds: Array):