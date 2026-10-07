import torch as t
import torch.nn as nn
from jaxtyping import Shaped
from einops import rearrange

from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.interfaces import MixerCaps, Residual
from torch_llm.core.batch_meta import BatchMeta
from torch_llm.model.rope import RoPE
from torch_llm.kernals.flash_attention import FlashAttentionFunction
from torch_llm.kernals.decode_attention import decode_attention_wrapper
from torch_llm.core.registry import MIXERS

'''
prefill
compute current K/V normally in BF16
→ use them normally for prefill attention
→ quantized cache append for future use

decode
compute new K/V
→ quantized append
→ paged decode kernel reads quantized history + scales


'''
@MIXERS.register("gqa")
class Attention(nn.Module):
    capabilities= MixerCaps(needs_positions=True)

    def __init__(self,
                 config,
                 layer_idx,
                 ):
        super().__init__()
        d_model = config.d_model
        num_q_heads = config.num_q_heads
        num_kv_heads = config.num_kv_heads
        head_dim = config.head_dim
        model_max_seq_len = config.model_max_seq_len
        theta = config.rope_theta

        self.layer_idx = layer_idx

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, d_model, bias=False)

        self.rope = RoPE(head_dim, model_max_seq_len, theta)

    def forward(
            self,
            state: Shaped[t.Tensor, "T d_model"],
            batch_meta: BatchMeta,
            paged_kv_cache = None,
            attention_mask=None,
        ) -> tuple[t.Tensor, AuxOutputs]:

        q = self.q_proj(state)
        k = self.k_proj(state)
        v = self.v_proj(state)
        q = rearrange(q, "T (h d) -> T h d", h=self.num_q_heads, d=self.head_dim)
        k = rearrange(k, "T (h d) -> T h d", h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(v, "T (h d) -> T h d", h=self.num_kv_heads, d=self.head_dim)

        b = batch_meta.cu_seqlens.shape[0] - 1
        q_rope = self.rope(q, batch_meta.token_positions)
        k_rope = self.rope(k, batch_meta.token_positions)

        if batch_meta.mode == 'train':
            assert paged_kv_cache is None
            attention_output = FlashAttentionFunction.apply(
                q_rope,
                k_rope,
                v,
                batch_meta.cu_seqlens,
                batch_meta.max_seqlen
            )

        elif batch_meta.mode == 'decode':
            assert paged_kv_cache is not None
            paged_kv_cache.append_kv(
                batch_meta.cache_context.physical_blocks,
                batch_meta.cache_context.block_offsets,
                k_rope,
                v,
            )

            attention_output = decode_attention_wrapper(
                q_rope,
                paged_kv_cache,
                batch_meta.cache_context
            )

        else:
            assert paged_kv_cache is not None
            attention_output = FlashAttentionFunction.apply(
                q_rope,
                k_rope,
                v,
                batch_meta.cu_seqlens,
                batch_meta.max_seqlen
            )

            paged_kv_cache.append_kv(
                batch_meta.cache_context.physical_blocks,
                batch_meta.cache_context.block_offsets,
                k_rope,
                v,
            )

        attention_output = rearrange(attention_output, 'T h d -> T (h d)')

        aux_outputs = AuxOutputs()

        return self.o_proj(attention_output), aux_outputs






