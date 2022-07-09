from typing import Dict, List, Tuple, Any
import numpy as np
import gym



class Missing_Obs_Wrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, no_obs_index: List[int]) -> None:
        super().__init__(env)
        self.no_obs_index = no_obs_index

    def _clear_observation(self, obs: np.array) -> np.array:
        for index in self.no_obs_index:
            obs[index] = 0.
        return obs

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        obs_, r, done, info = super().step(action)
        return self._clear_observation(obs_), r, done, info
    
    def reset(self) -> Any:
        obs = super().reset()
        return self._clear_observation(obs)