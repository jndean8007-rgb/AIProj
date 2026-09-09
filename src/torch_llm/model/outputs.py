from dataclasses import dataclass
import torch as t

@dataclass
class ModelOutput:
    logits: t.Tensor
    moe_stats: list
    aux_loss: t.Tensor