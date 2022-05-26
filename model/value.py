from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from state_filtration_for_qd.model.common import call_mlp


class VFunction(nn.Module):
    def __init__(self, o_dim: int, hidden_layers: List[int]) -> None:
        super(VFunction, self).__init__()
        self.o_dim = o_dim
        self.hidden_layers = hidden_layers
        self.model = call_mlp(
            o_dim,
            1,
            hidden_layers,
            inter_activation='Tanh',
            output_activation='Identity'
        )

    def __call__(self, obs: torch.tensor) -> torch.tensor:
        return self.model(obs)


class QFunction(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, hidden_layers: List[int]) -> None:
        super(QFunction, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.model = call_mlp(
            o_dim + a_dim,
            1,
            hidden_layers,
            'ReLU',
            'Identity'
        )

    def __call__(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        return self.model(torch.cat([obs, a], -1))


class TwinQFunction(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, hidden_layers: List[int]) -> None:
        super(TwinQFunction, self).__init__()
        self.Q1_model = QFunction(o_dim, a_dim, hidden_layers)
        self.Q2_model = QFunction(o_dim, a_dim, hidden_layers)

    def __call__(self, obs: torch.tensor, a: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        Q1_value, Q2_value = self.Q1_model(obs, a), self.Q2_model(obs, a)
        return Q1_value, Q2_value

    def call_Q1(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        return self.Q1_model(obs, a)
    
    def call_Q2(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        return self.Q2_model(obs, a)