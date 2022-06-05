from typing import Dict, List, Tuple
import numpy as np
from gym.envs.mujoco.ant import AntEnv


class Disturb_Dynamics_Ant(AntEnv):
    def __init__(self, leg_mass_scale: float, ankle_friction_scale: float):
        self.episode_length = 1000
        self.episode_step   = 0
        
        super().__init__()
        self.model.body_mass[2] = self.model.body_mass[2] * leg_mass_scale                         # four leg mass scale
        self.model.body_mass[4] = self.model.body_mass[4] * leg_mass_scale
        self.model.body_mass[6] = self.model.body_mass[6] * leg_mass_scale
        self.model.body_mass[8] = self.model.body_mass[8] * leg_mass_scale

        self.model.geom_friction[4][0] = self.model.geom_friction[4][0] * ankle_friction_scale       # ankle friction scale
        self.model.geom_friction[7][0] = self.model.geom_friction[7][0] * ankle_friction_scale 
        self.model.geom_friction[10][0] = self.model.geom_friction[10][0] * ankle_friction_scale 
        self.model.geom_friction[13][0] = self.model.geom_friction[13][0] * ankle_friction_scale 

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