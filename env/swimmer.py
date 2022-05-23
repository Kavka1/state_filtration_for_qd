from typing import List, Dict, Tuple
import numpy as np
from gym.envs.mujoco.swimmer import SwimmerEnv


MISSING_VEL_JOINT = ['rot', 'rot1', 'rot2']
MISSING_POS_COORD = ['2']


class Missing_Info_Swimmer(SwimmerEnv):
    def __init__(self, missing_obs_info: Dict, apply_missing_obs: bool = False):
        self.episode_length = 1000
        self.episode_step = 0
        self.missing_joint = missing_obs_info['missing_joint']
        self.missing_coord = missing_obs_info['missing_coord']
        self.apply_missing_obs = apply_missing_obs

        for joint in self.missing_joint:
            assert joint in MISSING_VEL_JOINT, f"Invalid missing joint {joint}"
        for coord in self.missing_coord:
            assert coord in MISSING_POS_COORD, f"Invalid missing coordinate {coord}"
        super().__init__()

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat[1:]
        qvel = np.clip(self.sim.data.qvel.flat, -10, 10)
        
        if self.apply_missing_obs:
            qvel = self._drop_infeasible_jnt_vel(qvel)
            qpos = self._drop_infeasible_coord_pos(qpos)
        
        return np.concatenate([
            qpos, 
            qvel
        ])

    def _drop_infeasible_jnt_vel(self, qvel: np.array) -> np.array:
        """
        Missing observation info of the joint velocity
        Original info: 
            qvel - velocity for 5 joints:
                slider1, slider2,
                rot, rot1, rot2,
        """
        feasible_q_vel = np.copy(qvel)
        rot_joint_index, rot_1_joint_index, rot_2_joint_index = [2], [3], [4]

        if 'rot' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, rot_joint_index)
            rot_1_joint_index = [rot_1_joint_index[i] - 1 for i in range(1)]
            rot_2_joint_index = [rot_2_joint_index[i] - 1 for i in range(1)]
        if 'rot1' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, rot_1_joint_index)
            rot_2_joint_index = [rot_2_joint_index[i] - 1 for i in range(1)]
        if 'rot2' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, rot_2_joint_index)

        return feasible_q_vel

    def _drop_infeasible_coord_pos(self, qpos: np.array) -> np.array:
        """
        Missing observation info of the coordinate position
        Original info:
            qpos - position in three generalized coordinates
                [0]: (x), (y), angle for generalized coordinate 1
                [1:2]: ? for generalized coordinate 2
        """
        feasible_q_pos = np.copy(qpos)
        coord_2_index = [1,2]
        if '2' in self.missing_coord:
            feasible_q_pos = np.delete(feasible_q_pos, coord_2_index)
        
        return feasible_q_pos

    def step(self, a):
        obs, r, done, info = super().step(a)
        if self.episode_step >= self.episode_length:
            done = True
        self.episode_step += 1
        return obs, r, done, info

    def reset(self):
        self.episode_step = 0
        return super().reset()

    def _decompose_qpos_qvel(self, obs: np.array) -> Tuple:
        """
        Factorize the observation to qpos and qvel
        """
        dim_qpos = len(self.sim.data.qpos.flat[2:])
        qpos = obs[:dim_qpos]
        qvel = obs[dim_qpos:]
        return qpos, qvel

    def _process_obs(self, obs: np.array) -> np.array:
        """
        Process the original obs to missing info version
        """
        qpos, qvel = self._decompose_qpos_qvel(obs)
        qvel = self._drop_infeasible_jnt_vel(qvel)
        qpos = self._drop_infeasible_coord_pos(qpos)
        return np.concatenate([qpos, qvel], axis=-1)

    @property
    def action_bound(self) -> float:
        return self.action_space.high[0]