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
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attn_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )

        self.attention = Attention(
            d_model=config.d_model,
            num_q_heads=config.num_q_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            model_max_seq_len=config.model_max_seq_len,
            theta=config.rope_theta
        )

        self.moe_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )

        self.moe = MoE(
            num_experts=config.num_experts,
            top_k=config.top_k,
            d_model=config.d_model,
            d_ff=config.d_ff,
            bias_lr=config.router_bias_lr,
            beta=config.router_beta
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
