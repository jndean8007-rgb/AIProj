import torch as t
import torch.nn as nn
from typing import Literal
from jaxtyping import Shaped

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
            cu_seqlens: Shaped[t.Tensor, 'B+1'],
            token_positions: Shaped[t.Tensor, 'T'],
            batch_max_seq_len: int,
            mode: Literal['train', 'prefill', 'decode'] = 'train',
            kv_cache=None
    ):

        attn_out = self.attention(
            self.attn_norm(x),
            token_positions,
            cu_seqlens,
            batch_max_seq_len,
            kv_cache=kv_cache,
            mode=mode,
        )

        x = x + attn_out

        moe_out, stats = self.moe(
            self.moe_norm(x),
        )

        output = x + moe_out

        return output, stats
