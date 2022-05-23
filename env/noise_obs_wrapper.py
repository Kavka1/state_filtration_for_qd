from typing import Dict, List, Tuple, Any
import numpy as np
import gym



class Noise_Obs_Wrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, noise_type: str, noise_scale: float, noise_index: List[int]) -> None:
        super().__init__(env)
        if noise_type == 'Gaussian':
            self.noise = np.random.normal
            self.noise_scale = noise_scale
        else:
            raise NotImplementedError(f'Invalid observation noise type {noise_type}')
        self.noise_index = noise_index

    def _disrupt_observation(self, obs: np.array) -> np.array:
        for index in self.noise_index:
            obs[index] += self.noise((0,), (self.noise_scale,))
        return obs

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        obs_, r, done, info = super().step(action)
        return self._disrupt_observation(obs_), r, done, info
    
    def reset(self) -> Any:
        obs = super().reset()
        return self._disrupt_observation(obs)