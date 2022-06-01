from typing import List
import numpy as np
import gym
from gym.envs.mujoco.hopper import HopperEnv


class Broken_Leg_Hopper(HopperEnv):
    def __init__(self, leg_jnt_scale: float, foot_jnt_scale: float):
        self.episode_length = 1000
        self.episode_step   = 0

        super().__init__()
        self.model.jnt_range[4][0] = self.model.jnt_range[4][0] * leg_jnt_scale     # leg joint minimum rotation
        self.model.jnt_range[4][1] = self.model.jnt_range[4][1] * leg_jnt_scale
        self.model.jnt_range[5][0] = self.model.jnt_range[5][0] * foot_jnt_scale    # foot joint
        self.model.jnt_range[5][1] = self.model.jnt_range[5][1] * foot_jnt_scale

    def step(self, a):
        obs, r, done, info = super().step(a)
        if self.episode_step >= self.episode_length:
            done = True
        self.episode_step += 1
        return obs, r, done, info

    def reset(self):
        self.episode_step = 0
        return super().reset()

    @property
    def action_bound(self) -> float:
        return self.action_space.high[0]