import torch as t
import torch.nn as nn

from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta
from torch_llm.core.registry import OUTPUT_HEADS
from torch_llm.model.rmsnorm import RMSNorm


@OUTPUT_HEADS.register("lm")
class LMHead(nn.Module):
    """
    Final norm + projection to vocabulary logits, with the projection weight
    tied to the token embedding.

    The head owns its norm so that every head (e.g. future MTP heads) normalizes
    its own input.

    Tying shares the embedding's Parameter object; the embedding module itself is
    not stored, so no extra checkpoint key (head.embedding.weight) is created.
    """

    def __init__(self, config, embedding: nn.Embedding):
        super().__init__()

        # Order matters: norm before proj keeps parameter order identical to the
        # old final_norm -> lm_head layout (saved optimizer state is positional).
        self.norm = RMSNorm(d_model=config.d_model, eps=config.rms_eps)
        self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.proj.weight = embedding.weight

    def forward(self, h: t.Tensor, meta: BatchMeta) -> tuple[t.Tensor, AuxOutputs]:
        """h: [T, D] final residual stream -> logits: [T, V]"""
        return self.proj(self.norm(h)), AuxOutputs()
