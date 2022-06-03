from typing import Dict, List, Tuple, Any
import numpy as np
import dmc2gym
import gym
from gym.core import _ActionType, _OperationType


class Missing_Info_Quadruped_Walk(gym.Env):
    def __init__(self, missing_obs_info: Dict, apply_missing_obs: bool = False) -> None:
        super().__init__()
        self._env = dmc2gym.make(
            domain_name='quadruped',
            task_name='walk',
            from_pixels= False,
            visualize_reward=True
        )
        self.missing_leg        =   missing_obs_info['missing_leg']
        self.apply_missing_obs  =   apply_missing_obs
    
    @property
    def action_space(self) -> gym.Box:
        return self._env.action_space
    
    @property
    def observation_space(self) -> gym.Box:
        return self._env.observation_space

    @property
    def action_bound(self) -> float:
        return self._env.action_space.high[0]

    def step(self, action: _ActionType) -> Tuple[_OperationType, float, bool, Dict[str, Any]]:
        obs, r, done, info = self._env.step(action)
        if self.apply_missing_obs:
            obs = self._process_obs(obs)
        return obs, r, done, info

    def reset(self) -> Any:
        obs = self._env.reset()
        if self.apply_missing_obs:
            obs = self._process_obs(obs)
        return obs
    
    def _process_obs(self, obs: np.array) -> np.array:
        """
        Process the original obs to missing info version
        """
        # todo: update this api
        obs_copy = np.copy(obs)
        return obs_copy



if __name__ == '__main__':
    env = Missing_Info_Quadruped_Walk(
        missing_obs_info={'missing_leg': []},
        apply_missing_obs= False
    )
    for _ in range(100):
        done = False
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            obs, r, done, _ = env.step(action)
    
    