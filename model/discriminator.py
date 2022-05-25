from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from state_filtration_for_qd.model.common import call_mlp


class DeltaSA_Z_Discrete_Discriminator(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, z_dim: int, hidden_layers: List[int]) -> None:
        super().__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers

        self.in_dim = self.o_dim + self.a_dim
        self.out_dim = self.z_dim
        self.model = call_mlp(
            self.in_dim,
            self.out_dim,
            self.hidden_layers,
            inter_activation= 'ReLU',
            output_activation= 'Identity'
        )

    def inference(self, obs: torch.tensor, obs_: torch.tensor, a: torch.tensor, z: torch.tensor) -> np.array:
        delta_obs = obs_ - obs
        x = torch.concat([delta_obs, a], dim=-1)
        logits = self.model(x)
        prob = torch.softmax(logits, -1)
        logprob = torch.log(prob[z] + 1e-6).detach().numpy()
        return logprob

    def __call__(self, obs: torch.tensor, obs_: torch.tensor, a: torch.tensor) -> torch.tensor:
        x = torch.concat([obs_ - obs, a], dim=-1)
        logits = self.model(x)
        return logits


class DeltaSZ_A_Discriminator(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, z_dim: int, hidden_layers: List[int], logstd_min: float, logstd_max: float) -> None:
        super().__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers

        self.in_dim = self.o_dim + self.z_dim
        self.out_dim = self.a_dim * 2
        self.model = call_mlp(
            self.in_dim,
            self.out_dim,
            self.hidden_layers,
            inter_activation= 'ReLU',
            output_activation= 'Identity'
        )
        self.logstd_min = nn.Parameter(torch.ones(size=(self.a_dim,)).float() * logstd_min) 
        self.logstd_max = nn.Parameter(torch.ones(size=(self.a_dim,)).float() * logstd_max)

    def __call__(self, obs: torch.tensor, obs_: torch.tensor, z: torch.tensor) -> torch.distributions.Distribution:
        x = torch.concat([obs_ - obs, z], dim=-1)
        x = self.model(x)
        mean, logstd = torch.chunk(x, 2, dim=-1)

        mean = torch.tanh(mean)
        logstd = self.logstd_max - F.softplus(self.logstd_max - logstd)
        logstd = self.logstd_min + F.softplus(logstd - self.logstd_min)

        return Normal(mean, torch.exp(logstd))


class S_DiscreteZ_Discriminator(nn.Module):
    def __init__(self, o_dim: int, z_dim: int, hidden_layers: List[int]) -> None:
        super().__init__()
        self.o_dim = o_dim
        self.z_dim = z_dim
        
        self.model = call_mlp(
            in_dim= self.o_dim,
            out_dim= self.z_dim,
            hidden_layers= hidden_layers,
            inter_activation='ReLU',
            output_activation='Identity'
        )

    def __call__(self, o: torch.tensor) -> torch.tensor:
        logits = self.model(o)
        return logits