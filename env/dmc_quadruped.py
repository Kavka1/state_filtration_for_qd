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