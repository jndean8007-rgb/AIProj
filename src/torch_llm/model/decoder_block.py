import torch as t
import torch.nn as nn
from jaxtyping import Shaped

from core.registry import MIXERS, FFNS
from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta
from torch_llm.model.model_config import ModelConfig
from torch_llm.model.rmsnorm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        layer_spec = config.layers[layer_idx]

        # memory = TOKEN_MEMORIES.get(layer_spec.memory)

        self.mixer = MIXERS.get(layer_spec.mixer)(
            config=config,
            layer_idx=layer_idx,
        )
        self.ffn = FFNS.get(layer_spec.ffn)(
            config=config,
            layer_idx=layer_idx,
        )

        self.mixer_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )

        self.ffn_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )


    def forward(
            self,
            x: Shaped[t.Tensor, 'T D'],
            batch_meta: BatchMeta,
            paged_kv_cache = None,
    ):

        attn_out = self.mixer(
            self.attn_norm(x),
            batch_meta=batch_meta,
            paged_kv_cache=paged_kv_cache,
        )

        x = x + attn_out

        aux_outputs = AuxOutputs()

        moe_out, new_aux_outputs = self.ffn(
            self.ffn_norm(x),
        )

        output = x + moe_out

        aux_outputs = aux_outputs.merged(new_aux_outputs, prefix="ffn")
        return output, aux_outputs
