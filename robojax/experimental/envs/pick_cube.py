from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from mani_skill2.envs.pick_and_place.base_env import StationaryManipulationEnv

@register_env("PickCube-v1", max_episode_steps=200)
class PickCubeEnv(StationaryManipulationEnv):
    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, obj_init_rot_z=True, reward_config=dict(static_reward=True, stage_scaler=2, grasp_reward=True), **kwargs):
        self.reward_config=reward_config
        self.obj_init_rot_z = obj_init_rot_z
        self.cube_half_size = np.array([0.02] * 3, np.float32)
        super().__init__(*args, **kwargs)

        self.max_reward = 0

        self.staged_reward_weights = [1, 1, 1]
        for i in range(3):
            self.staged_reward_weights[i] = (1 + i * self.reward_config["stage_scaler"])
            self.max_reward += self.staged_reward_weights[i]
        self.max_reward += 2

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        xyz = np.hstack([xy, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
        self.obj.set_pose(Pose(xyz, q))

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.obj.pose.p

        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.5) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - obj_pos) > self.min_goal_dist:
                if verbose:
                    print(f"Found a valid goal at {i}-th trial")
                break

        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
                obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
            )
        return obs

    def check_obj_placed(self):
        return np.linalg.norm(self.goal_pos - self.obj.pose.p) <= self.goal_thresh

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        is_robot_static = self.check_robot_static()
        return dict(
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed and is_robot_static,
        )
    
    def compute_dense_reward(self, info, **kwargs):
        """
        stage_scaler multiplies stage i by (1 + stage_scaler * i)
        
        max reward is 5 (on success)
        """
        reward = 0.0
        

        if info["success"]:
            reward += self.max_reward
        else:
            tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
            tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
            reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
            reward += reaching_reward * self.staged_reward_weights[0]

            is_grasped = self.agent.check_grasp(self.obj)
            if self.reward_config['grasp_reward']:
                reward += 1 if is_grasped else 0.0

            if is_grasped:
                
                obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
                place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
                reward += place_reward * self.staged_reward_weights[1]

                # static reward
                if self.reward_config['static_reward']:
                    if self.check_obj_placed():
                        qvel = self.agent.robot.get_qvel()[:-2]
                        static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                        reward += static_reward * self.staged_reward_weights[2]
        if self.reward_config['scale_reward']:
            reward = reward / self.max_reward
        return reward

    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            self.goal_site.unhide_visual()
            ret = super().render(mode=mode)
            self.goal_site.hide_visual()
        else:
            ret = super().render(mode=mode)
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])
