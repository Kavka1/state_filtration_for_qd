from typing import Any, Dict, List
import torch
import torch.nn as nn
from torch.distributions import Normal

from state_filtration_for_qd.model.common import call_mlp


class FixStdPolicy_Ensemble(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, num_policies: int ,hidden_layers: List[int], action_std: float, activation: str) -> None:
        super(FixStdPolicy_Ensemble, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.num_policies = num_policies
        self.hidden_layers = hidden_layers
        self.policies = [
            call_mlp(
                o_dim,
                a_dim,
                hidden_layers,
                inter_activation=activation,
                output_activation='Tanh'
            )
            for _ in range(num_policies)
        ]
        self.ac_std = nn.Parameter(torch.ones(size=(a_dim,)) * action_std, requires_grad=False)

    def act(self, obs: torch.tensor, z: int, with_noise: True) -> torch.tensor:
        with torch.no_grad():
            if with_noise:
                mean = self.policies[z](obs)
                dist = Normal(mean, self.ac_std)
                action = dist.sample()
            else:
                action = self.policies[z](obs)
        return action

    def __call__(self, obs: torch.tensor, z: Any[int, torch.tensor]) -> torch.distributions.Distribution:
        mean = self.policies[z](obs)
        dist = Normal(mean, self.ac_std)
        return dist

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"| - Loaded model from {path} - |")
