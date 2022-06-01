from typing import Dict, List, Tuple
import numpy as np
import gym
from gym.envs.mujoco.walker2d import Walker2dEnv


class Broken_Leg_Walker(Walker2dEnv):
    def __init__(self, is_left_leg: bool, leg_jnt_scale: float, foot_jnt_scale: float):
        self.episode_length = 1000
        self.episode_step   = 0
        self.is_left_leg    = is_left_leg
        
        if self.is_left_leg:
            self.model.jnt_range[7][0] = self.model.jnt_range[7][0] * leg_jnt_scale     # left leg joint minimum rotation
            self.model.jnt_range[8][0] = self.model.jnt_range[8][0] * foot_jnt_scale    # left foot joint minimum rotation
        else:
            self.model.jnt_range[4][0] = self.model.jnt_range[4][0] * leg_jnt_scale     # right leg joint
            self.model.jnt_range[5][0] = self.model.jnt_range[5][0] * foot_jnt_scale    # right foot joint 
        super().__init__()

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