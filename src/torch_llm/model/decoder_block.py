import torch as t
import torch.nn as nn
from jaxtyping import Shaped

from torch_llm.core.interfaces import Residual
from torch_llm.core.registry import MIXERS, FFNS
from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta
from torch_llm.model.model_config import ModelConfig
from torch_llm.model.rmsnorm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()

        self.sublayer_start = config.sublayer_indices[layer_idx]

        layer_spec = config.layers[layer_idx]

        # memory = TOKEN_MEMORIES.get(layer_spec.memory)
        self.mixer_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )

        self.mixer = MIXERS.get(layer_spec.mixer)(
            config=config,
            layer_idx=layer_idx,
        )

        self.ffn_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )

        self.ffn = FFNS.get(layer_spec.ffn)(
            config=config,
            layer_idx=layer_idx,
        )
    def forward(
            self,
            state: Shaped[t.Tensor, 'T D'],
            residual: Residual,
            batch_meta: BatchMeta,
            paged_kv_cache = None,
    ):
        aux_outputs = AuxOutputs()

        update, mixer_aux = self.mixer(
            state=self.mixer_norm(residual.read(state, self.sublayer_start)),
            batch_meta=batch_meta,
            paged_kv_cache=paged_kv_cache,
        )

        aux_outputs = aux_outputs.merged(mixer_aux, prefix="mixer")

        state = residual.write(state, self.sublayer_start, update)


        moe_out, ffn_aux = self.ffn(
            state=self.ffn_norm(residual.read(state, self.sublayer_start + 1)),
            batch_meta=batch_meta,
        )

        state = residual.write(state, self.sublayer_start + 1, moe_out)

        aux_outputs = aux_outputs.merged(ffn_aux, prefix="ffn")
        return state, aux_outputs
