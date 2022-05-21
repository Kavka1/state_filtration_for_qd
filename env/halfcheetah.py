from typing import List, Dict, Tuple
import numpy as np
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


MISSING_VEL_JOINT = ['thigh', 'shin', 'foot']
MISSING_POS_COORD = ['2', '3']


class Missing_Info_HalfCheetah(HalfCheetahEnv):
    def __init__(self, missing_obs_info: Dict):
        self.episode_length = 1000
        self.episode_step = 0
        self.missing_joint = missing_obs_info['missing_joint']
        self.missing_coord = missing_obs_info['missing_coord']

        for joint in self.missing_joint:
            assert joint in MISSING_VEL_JOINT, f"Invalid missing joint {joint}"
        for coord in self.missing_coord:
            assert coord in MISSING_POS_COORD, f"Invalid missing coordinate {coord}"
        super().__init__()

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat[1:]
        qvel = self.sim.data.qvel.flat

        #feasible_q_vel = self._drop_infeasible_jnt_vel(qvel)
        #feasible_q_pos = self._drop_infeasible_coord_pos(qpos)
        
        return np.concatenate([
            qpos, 
            qvel
        ])

    def _drop_infeasible_jnt_vel(self, qvel: np.array) -> np.array:
        """
        Missing observation info of the joint velocity
        Original info: 
            qvel - velocity for 9 joints:
                rootx, rooty, rootz,
                bthigh, bshin, bfoot,
                fthigh, fshin, ffoot,
        """
        feasible_q_vel = np.copy(qvel)
        thigh_joint_index, shin_joint_index, foot_joint_index = [3,6], [4,7], [5,8]

        if 'thigh' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, thigh_joint_index)
            shin_joint_index = [shin_joint_index[i] - i - 1 for i in range(2)]
            foot_joint_index = [foot_joint_index[i] - i - 1 for i in range(2)]
        if 'shin' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, shin_joint_index)
            foot_joint_index = [foot_joint_index[i] - i - 1 for i in range(2)]
        if 'foot' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, foot_joint_index)

        return feasible_q_vel

    def _drop_infeasible_coord_pos(self, qpos: np.array) -> np.array:
        """
        Missing observation info of the coordinate position
        Original info:
            qpos - position in three generalized coordinates
                [0:1]: (x), y, angle for generalized coordinate 1
                [2:4]: x, y, angle for generalized coordinate 2
                [5:7]: x, y, angle for generalized coordinate 3
        """
        feasible_q_pos = np.copy(qpos)
        coord_2_index, coord_3_index = [2,3,4], [5,6,7]
        if '3' in self.missing_coord:
            feasible_q_pos = np.delete(feasible_q_pos, coord_3_index)
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
        dim_qpos = len(self.sim.data.qpos.flat[1:])
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