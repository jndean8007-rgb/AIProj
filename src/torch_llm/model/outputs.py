from dataclasses import dataclass
import torch as t

from torch_llm.core.aux_outputs import AuxOutputs


@dataclass
class ModelOutputs:
    logits: t.Tensor
    aux_outputs: AuxOutputs

    def __post_init__(self):
        if not all(aux_loss.device == self.logits.device for aux_loss in self.aux_outputs.losses.values()):
            raise ValueError("Device of Auxillary loss output and logits mus be the same.")