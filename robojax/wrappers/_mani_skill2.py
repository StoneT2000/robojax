import gym
import gymnasium
import gymnasium.spaces as spaces
import numpy as np
from mani_skill2.envs.sapien_env import BaseEnv


class ManiSkill2Wrapper(gymnasium.Wrapper):
    def __init__(self, env: gym.Env, render_mode: str = "rgb_array"):
        super().__init__(env)
        self.wrapper_render_mode = render_mode
        self._action_space = spaces.Box(
            env.action_space.low,
            env.action_space.high,
            env.action_space.shape,
            env.action_space.dtype,
        )
        self._observation_space = spaces.Box(
            env.observation_space.low,
            env.observation_space.high,
            env.observation_space.shape,
            env.observation_space.dtype,
        )

    def reset(self, *, seed=None, options=None):
        self.env: BaseEnv
        if seed is not None:
            self.env.seed(
                seed
            )  # this call is necessary as maniskill maintains a seed generator per episode that defaults to a fixed value unless seed
        obs = self.env.reset(seed=seed)
        return obs, {}

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info

    @property
    def render_mode(self):
        return self.wrapper_render_mode

    def render(self):
        return self.env.render(self.wrapper_render_mode)


class PegInsertionSideStats(gymnasium.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.grasp_count = 0
        self.rotated_properly_count = 0
        self.off_the_ground_count = 0
        self.success_once = False

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.grasp_count = 0
        self.rotated_properly_count = 0
        self.off_the_ground_count = 0
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # add useful stats to aggregate
        # self.env: PegInsertionSideEnv
        peg = self.env.peg

        grasp_rot_loss = self.env.grasp_loss()
        rotated_properly = grasp_rot_loss < 0.2

        self.rotated_properly_count += int(rotated_properly)
        self.grasp_count += self.env.agent.check_grasp(peg, max_angle=20)
        off_the_ground = peg.pose.p[-1] > 0.05
        self.off_the_ground_count += int(off_the_ground)

        peg_head_pos_at_hole = info["peg_head_pos_at_hole"]
        x_flag = -0.015 <= peg_head_pos_at_hole[0]
        y_flag = -self.env.box_hole_radius <= peg_head_pos_at_hole[1] <= self.env.box_hole_radius
        z_flag = -self.env.box_hole_radius <= peg_head_pos_at_hole[2] <= self.env.box_hole_radius
        peg_goal_x_dist = peg_head_pos_at_hole[0] - (-0.015)
        peg_goal_y_dist = abs(peg_head_pos_at_hole[1]) - self.env.box_hole_radius
        peg_goal_z_dist = abs(peg_head_pos_at_hole[2]) - self.env.box_hole_radius
        self.success_once = self.success_once | info["success"]
        info["stats"] = dict(
            grasp_count=self.grasp_count,
            rotated_properly_count=self.rotated_properly_count,
            success_at_end=int(info["success"]),
            success=self.success_once,
            off_the_ground_count=self.off_the_ground_count,
            peg_goal_x_dist=peg_goal_x_dist,
            peg_goal_y_dist=peg_goal_y_dist,
            peg_goal_z_dist=peg_goal_z_dist,
        )
        return observation, reward, terminated, truncated, info


class MS2Stats(gymnasium.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.success_once = False

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # add useful stats to aggregate
        self.success_once = self.success_once | info["success"]
        info["stats"] = dict(
            success_at_end=int(info["success"]),
            success=self.success_once,
        )
        return observation, reward, terminated, truncated, info


class PickCubeStats(gymnasium.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.grasp_count = 0
        self.obj_placed_count = 0
        self.off_the_ground_count = 0
        self.success_once = False

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.grasp_count = 0
        self.obj_placed_count = 0
        self.off_the_ground_count = 0
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # add useful stats to aggregate
        obj_to_goal_dist = np.linalg.norm(self.env.goal_pos - self.env.obj.pose.p)
        self.obj_placed_count += int(info["is_obj_placed"])
        self.grasp_count += self.env.agent.check_grasp(self.env.obj, max_angle=30)
        off_the_ground = self.env.obj.pose.p[-1] > 0.05
        self.off_the_ground_count += int(off_the_ground)
        self.success_once = self.success_once | info["success"]
        info["stats"] = dict(
            grasp_count=self.grasp_count,
            obj_placed_count=self.obj_placed_count,
            obj_to_goal_dist=obj_to_goal_dist,
            success_at_end=int(info["success"]),
            success=self.success_once,
            off_the_ground_count=self.off_the_ground_count,
        )
        return observation, reward, terminated, truncated, info


class ContinuousTaskWrapper(gymnasium.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        terminated = False
        return observation, reward, terminated, truncated, info
