import torch as t
import torch.nn as nn
from jaxtyping import Shaped

from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta
from torch_llm.model.model_config import ModelConfig
from torch_llm.model.moe.moe import MoE
from torch_llm.model.rmsnorm import RMSNorm
from torch_llm.model.attention import Attention


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )

        self.attention = Attention(
            config=config,
            layer_idx=layer_idx,
        )

        self.moe_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )

        self.moe = MoE(
            config=config,
            layer_idx=layer_idx,
        )


    def forward(
            self,
            x: Shaped[t.Tensor, 'T D'],
            batch_meta: BatchMeta,
            paged_kv_cache = None,
    ):

        attn_out = self.attention(
            self.attn_norm(x),
            batch_meta=batch_meta,
            paged_kv_cache=paged_kv_cache,
        )

        x = x + attn_out

        aux_outputs = AuxOutputs()

        moe_out, new_aux_outputs = self.moe(
            self.moe_norm(x),
        )

        output = x + moe_out

        aux_outputs = aux_outputs.merged(new_aux_outputs, prefix="ffn")
        return output, aux_outputs
