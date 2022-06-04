from typing import Dict, List, Tuple, Any
import numpy as np
import dmc2gym
import gym

class Missing_Info_Quadruped_Walk(object):
    def __init__(self, missing_obs_info: Dict, apply_missing_obs: bool = False) -> None:
        super().__init__()
        self._env = dmc2gym.make(
            domain_name='quadruped',
            task_name='walk',
            from_pixels= False,
            visualize_reward=True
        )
        self.missing_torso_vel  =   missing_obs_info['missing_torso_vel']
        self.missing_imu        =   missing_obs_info['missing_imu']
        self.apply_missing_obs  =   apply_missing_obs
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return self._env.action_space
    
    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._env.observation_space

    @property
    def action_bound(self) -> float:
        return self._env.action_space.high[0]

    def seed(self, seed) -> None:
        self._env.seed(seed)

    def render(self):
        return self._env.render('rgb_array')

    def step(self, action):
        obs, r, done, info = self._env.step(action)
        if self.apply_missing_obs:
            obs = self._process_obs(obs)
        return obs, r, done, info

    def reset(self) -> Any:
        obs = self._env.reset()
        if self.apply_missing_obs:
            obs = self._process_obs(obs)
        return obs
    
    def _drop_torso_vel(self, obs: np.array) -> np.array:
        """
        Drop the velocity information of torso
        """
        if self.missing_torso_vel[0]:
            obs = np.delete(obs, [44,45,46])
            return obs
        else:
            return obs

    def _drop_imu(self, obs: np.array) -> np.array:
        """
        Drop the imu sensor information 
        """
        if self.missing_imu[0]:
            obs = np.delete(obs, [48,49,50,51,52,53])
            return obs
        else:
            return obs

    def _process_obs(self, obs: np.array) -> np.array:
        """
        Process the original obs to missing info version
        """
        obs = self._drop_imu(obs)
        obs = self._drop_torso_vel(obs)
        return obs



if __name__ == '__main__':
    env = Missing_Info_Quadruped_Walk(
        missing_obs_info={'missing_torso_vel': [True], 'missing_imu': [True]},
        apply_missing_obs= True
    )
    for _ in range(100):
        done = False
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            obs, r, done, _ = env.step(action)
    
    