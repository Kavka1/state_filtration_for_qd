from typing import Dict, List, Tuple
import numpy as np
import gym
from gym.envs.mujoco.walker2d import Walker2dEnv


class Disturb_Dynamics_Walker(Walker2dEnv):
    def __init__(self, foot_mass_scale: float, foot_friction_scale: float):
        self.episode_length = 1000
        self.episode_step   = 0
        
        super().__init__()
        self.model.body_mass[7] = self.model.body_mass[7] * foot_mass_scale                         # two foot mass scale
        self.model.body_mass[4] = self.model.body_mass[4] * foot_mass_scale     
        self.model.geom_friction[7][0] = self.model.geom_friction[7][0] * foot_friction_scale       # foot friction scale
        self.model.geom_friction[4][0] = self.model.geom_friction[4][0] * foot_friction_scale 

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