from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from state_filtration_for_qd.model.common import call_mlp


class FixStdGaussianPolicy(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, hidden_layers: List[int], action_std: float, activation: str) -> None:
        super(FixStdGaussianPolicy, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.hidden_layers = hidden_layers
        self.model = call_mlp(
            o_dim,
            a_dim,
            hidden_layers,
            inter_activation=activation,
            output_activation='Tanh'
        )
        self.ac_std = nn.Parameter(torch.ones(size=(a_dim,)) * action_std, requires_grad=False)

    def act(self, obs: torch.tensor, with_noise: True) -> torch.tensor:
        with torch.no_grad():
            if with_noise:
                mean = self.model(obs)
                dist = Normal(mean, self.ac_std)
                action = dist.sample()
            else:
                action = self.model(obs)
        return action

    def __call__(self, obs: torch.tensor) -> torch.distributions.Distribution:
        mean = self.model(obs)
        dist = Normal(mean, self.ac_std)
        return dist

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"| - Loaded model from {path} - |")


class DiagGaussianPolicy(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, hidden_layers: List[int], logstd_min: float, logstd_max: float, activation: str = 'ReLU') -> None:
        super(DiagGaussianPolicy, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.hidden_layers = hidden_layers
        self.model = call_mlp(
            o_dim,
            a_dim * 2,
            hidden_layers,
            inter_activation=activation,
            output_activation='Identity'
        )
        self.logstd_min = nn.Parameter(torch.ones(size=(a_dim,)) * logstd_min, requires_grad=False)
        self.logstd_max = nn.Parameter(torch.ones(size=(a_dim,)) * logstd_max, requires_grad=False)
    
    def act(self, obs: torch.tensor, with_noise: True) -> torch.tensor:
        with torch.no_grad():
            x = self.model(obs)        
            mean, log_std = torch.chunk(x, 2, dim=-1)
        if with_noise:
            log_std = torch.clamp(log_std, self.logstd_min, self.logstd_max)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
        else:
            action = mean
        return torch.tanh(action)

    def __call__(self, obs: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.distributions.Distribution]:
        x = self.model(obs)
        mean, logstd = torch.chunk(x, 2, dim=-1)

        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd)

        dist = Normal(mean, std)
        arctanh_action = dist.rsample()
        action = torch.tanh(arctanh_action)

        logprob = dist.log_prob(arctanh_action).sum(dim=-1, keepdim=True)
        squashed_correction = torch.log(1 - action**2 + 1e-6).sum(dim=-1, keepdim=True)
        logprob = logprob - squashed_correction
        
        return action, logprob, dist

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        print(f"| - Loaded model from {path} - |")