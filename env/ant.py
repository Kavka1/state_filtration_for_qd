from typing import List, Tuple, Dict
import numpy as np
from gym.envs.mujoco.ant import AntEnv


MISSING_VEL_JOINT = ['torso', 'hip', 'ankle']
MISSING_POS_COORD = ['2', '3', '4', '5']


class Missing_Info_Ant(AntEnv):
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
        """
        Original Observation
            qpos - position in 5 generalized coordinates:
                [0]: (x), (y), angle for generalized coordinate 1
                [1:3]: x, y, angle for generalized coordinate 2
                [4:6]: x, y, angle for generalized coordinate 3
                [7:9]: x, y, angle for generalized coordinate 4
                [10:12]: x, y, angle for generalized coordinate 5
            qvel - velocity for 1 (free) + 8 (hinge) joints: 14 dim
                [0:5]: linear vel (3) + angular vel (3) for root joint: 
                [6:7]: hip_1 and ankle_1 of the front left leg 
                [8:9]: hip_2 and ankle_2 of the front right leg
                [10:11]: hip_3 and ankle_3 of the back left leg
                [12:13]: hip_4 and ankle_4 of the back right leg
        """
        qpos = self.sim.data.qpos.flat[2:]
        qvel = self.sim.data.qvel.flat
        cfrc_ext = np.clip(self.sim.data.cfrc_ext, -1, 1).flat

        if self.apply_missing_obs:
            qpos = self._drop_infeasible_coord_pos(qpos)
            qvel = self._drop_infeasible_jnt_vel(qvel)

        return np.concatenate([
            qpos,
            qvel,
            cfrc_ext
        ])
    
    def _drop_infeasible_jnt_vel(self, qvel: np.array) -> np.array:
        """
        Missing observation info of the joint velocity
        Original info:
            qvel - velocity for 1 (free) + 8 (hinge) joints: 14 dim
                linear vel (3) + angular vel (3) for root joint: [0:5]
                hip_1 and ankle_1 of the front left leg : [6:7]
                hip_2 and ankle_2 of the front right leg: [8:9]
                hip_3 and ankle_3 of the back left leg: [10:11]
                hip_4 and ankle_4 of the back right leg: [12:13]
        """
        feasible_q_vel = np.copy(qvel)
        torso_joint_index = [0,1,2,3,4,5]
        hip_joint_index = [6,8,10,12]
        ankle_joint_index = [7,9,11,13]
        
        if 'torso' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, torso_joint_index)
            hip_joint_index = [hip_joint_index[i] - 6 for i in range(4)]
            ankle_joint_index = [ankle_joint_index[i] - 6 for i in range(4)]
        if 'hip' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, hip_joint_index)
            ankle_joint_index = [ankle_joint_index[i] - i - 1 for i in range(4)]
        if 'ankle' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, ankle_joint_index)

        return feasible_q_vel

    def _drop_infeasible_coord_pos(self, qpos: np.array) -> np.array:
        """
        Missing observation info of the coordinate position
        Original info:
            qpos - position in 5 generalized coordinates:
                [0]: (x), (y), angle for generalized coordinate 1
                [1:3]: x, y, angle for generalized coordinate 2
                [4:6]: x, y, angle for generalized coordinate 3
                [7:9]: x, y, angle for generalized coordinate 4
                [10:12]: x, y, angle for generalized coordinate 5
        """
        feasible_q_pos = np.copy(qpos)
        coord_2_index = [1,2,3]
        coord_3_index = [4,5,6]
        coord_4_index = [7,8,9]
        coord_5_index = [10,11,12]
        if '5' in self.missing_coord:
            feasible_q_pos = np.delete(feasible_q_pos, coord_5_index)
        if '4' in self.missing_coord:
            feasible_q_pos = np.delete(feasible_q_pos, coord_4_index)
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
        Factorize the observation to qpos, qvel and cfrc_ext
        """
        dim_qpos = len(self.sim.data.qpos.flat[2:])
        dim_qvel = len(self.sim.data.qvel.flat)
        qpos = obs[:dim_qpos]
        qvel = obs[dim_qpos:dim_qpos + dim_qvel]
        cfrc_ext = obs[dim_qpos + dim_qvel:]
        return qpos, qvel, cfrc_ext

    def _process_obs(self, obs: np.array) -> np.array:
        """
        Process the original obs to missing info version
        """
        qpos, qvel, cfrc_ext = self._decompose_qpos_qvel(obs)
        qvel = self._drop_infeasible_jnt_vel(qvel)
        qpos = self._drop_infeasible_coord_pos(qpos)
        return np.concatenate([qpos, qvel, cfrc_ext], axis=-1)

    @property
    def action_bound(self) -> float:
        return self.action_space.high[0]