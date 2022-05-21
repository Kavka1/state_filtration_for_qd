from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.model.dynamics import DiagGaussianIDM



class Primitive(object):
    def __init__(self, model_config: Dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.policy = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std']
        ).to(device)
        self.inverse_model = DiagGaussianIDM(
            model_config['filtrated_o_dim'],
            model_config['a_dim'],
            model_config['idm_hidden_layers'],
            model_config['idm_logstd_min'],
            model_config['idm_logstd_max']
        ).to(device)

    def inference_action(self, obs_filt: np.array, next_obs_filt: np.array) -> torch.distributions:
        a_dist = self.inverse_model(
            torch.from_numpy(obs_filt).float().to(self.device),
            torch.from_numpy(next_obs_filt).float().to(self.device)
        )
        return a_dist
    
    def decision(self, obs: np.array, with_noise: bool = True) -> np.array:
        return self.policy.act(
            torch.from_numpy(obs).to(self.device).float(),
            with_noise
        ).detach().cpu().numpy()