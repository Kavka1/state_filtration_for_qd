from typing import Dict, List, Tuple
import numpy as np
import gym
from gym.envs.mujoco.ant import AntEnv


class Broken_Leg_Ant(AntEnv):
    def __init__(self, broken_legs: List, hip_jnt_scale: float, ankle_jnt_scale: float):
        self.episode_length = 1000
        self.episode_step   = 0
        self.broken_legs = broken_legs
        
        super().__init__()
        if '1' in self.broken_legs:
            self.model.jnt_range[1][0] = self.model.jnt_range[1][0] * hip_jnt_scale         # leg 1 hip joint minimum rotation
            self.model.jnt_range[1][1] = self.model.jnt_range[1][1] * hip_jnt_scale
            self.model.jnt_range[2][0] = self.model.jnt_range[2][0] * ankle_jnt_scale       # leg 1 ankle joint minimum rotation
            self.model.jnt_range[2][1] = self.model.jnt_range[2][1] * ankle_jnt_scale 
        if '2' in self.broken_legs:
            self.model.jnt_range[3][0] = self.model.jnt_range[3][0] * hip_jnt_scale         # leg 2 hip joint minimum rotation
            self.model.jnt_range[3][1] = self.model.jnt_range[3][1] * hip_jnt_scale
            self.model.jnt_range[4][0] = self.model.jnt_range[4][0] * ankle_jnt_scale       # leg 2 ankle joint minimum rotation
            self.model.jnt_range[4][1] = self.model.jnt_range[4][1] * ankle_jnt_scale 
        if '3' in self.broken_legs:
            self.model.jnt_range[5][0] = self.model.jnt_range[5][0] * hip_jnt_scale         # leg 3 hip joint minimum rotation
            self.model.jnt_range[5][1] = self.model.jnt_range[5][1] * hip_jnt_scale
            self.model.jnt_range[6][0] = self.model.jnt_range[6][0] * ankle_jnt_scale       # leg 3 ankle joint minimum rotation
            self.model.jnt_range[6][1] = self.model.jnt_range[6][1] * ankle_jnt_scale 
        if '4' in self.broken_legs:
            self.model.jnt_range[7][0] = self.model.jnt_range[7][0] * hip_jnt_scale         # leg 4 hip joint minimum rotation
            self.model.jnt_range[7][1] = self.model.jnt_range[7][1] * hip_jnt_scale
            self.model.jnt_range[8][0] = self.model.jnt_range[8][0] * ankle_jnt_scale       # leg 4 ankle joint minimum rotation
            self.model.jnt_range[8][1] = self.model.jnt_range[8][1] * ankle_jnt_scale 

    def step(self, action: np.array) -> Tuple:
        obs, r, done, info = super().step(action)
        if self.episode_step >= self.episode_length:
            done = True
        self.episode_step += 1
        return obs, r, done, info

    def reset(self) -> np.array:
        self.episode_step = 0
        return super().reset()

    @property
    def action_bound(self) -> float:
        return self.action_space.high[0]