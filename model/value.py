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