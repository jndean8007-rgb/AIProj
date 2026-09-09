import torch as t
import torch.nn as nn
from torch.nn.functional import linear
import torch.nn.functional as F
from jaxtyping import Shaped
import math

class ExpertWeights(nn.Module):
    def __init__(self,
                 num_experts,
                 d_model,
                 d_ff
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff

        self.gate_weight = nn.Parameter(
            t.empty(num_experts, d_ff, d_model)
        ) # follows pytorch linear semantics (each row is the weight vector for one output feature- ie d_ff weight for each d_model feature)

        self.up_weight =nn.Parameter(
            t.empty(num_experts, d_ff, d_model)
        )

        self.down_weight = nn.Parameter(
            t.empty(num_experts, d_model, d_ff)
        )

        nn.init.normal_(self.gate_weight, mean=0.0, std=1 / math.sqrt(d_model))
        nn.init.normal_(self.up_weight, mean=0.0, std=1 / math.sqrt(d_model))
        nn.init.normal_(self.down_weight, mean=0.0, std=1 / math.sqrt(d_ff))