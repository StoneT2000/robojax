"""
Tools working with pointclouds
"""

from chex import Array
import jax
import jax.numpy as jnp

@jax.jit
def depth_to_pcd(depth: Array, intrinsic: Array):
    """
    Lifts a 3d pointcloud from a depth image

    Returns a HxWx3 structured pointcloud

    Args :
        depth : HxW array representing the depth at each pixel
        intrinsic: 3x3 array representing the camera's intrinsic matrix.
    """
    v, u = jnp.indices(depth.shape)
    uv1 = jnp.stack([u + 0.5, v + 0.5, jnp.ones_like(depth)], axis=-1)
    points_viewer = uv1 @ jnp.linalg.inv(intrinsic).T * depth[..., None]  # [H, W, 3]
    return points_viewer

@jax.jit
def transform_pcd(pcd, mat):
    """
    Transforms a structured or unstructured pointcloud with a transformation matrix

    Args :
        pcd : the input point cloud of shape Nx3 for unstructured or HxWx3 for structured
        mat : the 4x4 rigid transformation matrix
    """
    pcd_is_structured = len(pcd.shape) == 3
    def _transform_structured_pcd(pcd):
        padding = ((0, 0), (0, 0), (0, 1))
        homogen_points = jnp.pad(pcd, padding,
                                'constant', constant_values=1)
        for i in range(3):
            s = jnp.sum(mat[i, :] * homogen_points, axis=-1)
            pcd.at[Ellipsis, i].set(s)
        return pcd
    def _transform_unstructured_pcd(pcd):
        return pcd @ mat[:3, :3].T + mat[:3, 3]
    return jax.lax.cond(pcd_is_structured, _transform_structured_pcd, _transform_unstructured_pcd, pcd)
    