from typing import Dict, List, Tuple
import numpy as np
import gym
from gym.envs.mujoco.hopper import HopperEnv


class Disturb_Dynamics_Hopper(HopperEnv):
    def __init__(self, foot_mass_scale: float, foot_friction_scale: float):
        self.episode_length = 1000
        self.episode_step   = 0
        
        super().__init__()
        self.model.body_mass[-1] = self.model.body_mass[-1] * foot_mass_scale                       # foot mass scale
        self.model.geom_friction[-1][0] = self.model.geom_friction[-1][0] * foot_friction_scale     # foot friction scale

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