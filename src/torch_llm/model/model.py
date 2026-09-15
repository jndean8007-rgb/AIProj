import torch.nn as nn
import torch as t
from typing_extensions import Literal
from jaxtyping import Shaped

from torch_llm.model.model_config import ModelConfig
from torch_llm.model.decoder_block import DecoderBlock
from torch_llm.model.outputs import ModelOutput
from torch_llm.model.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    def __init__(
            self,
            config:ModelConfig,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_eps
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        #potentially temporary weight tying
        self.lm_head.weight = self.embedding.weight

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(
            self,
            token_ids: Shaped[t.Tensor, 'T'],
            cu_seqlens: Shaped[t.Tensor, 'B+1'],
            token_positions: Shaped[t.Tensor, 'T'],
            batch_max_seq_len: int,
            kv_caches=None, # at model level length = num layers
            cache_slots=None,
            mode: Literal['train', 'prefill', 'decode'] = 'train',
    ):

        if cache_slots is not None:
            cache_slots = t.tensor(cache_slots, dtype=t.long)

        moe_stats = []

        x = self.embedding(token_ids)

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if kv_caches is None else kv_caches[layer_idx]

            x, stats= block(
                x=x,
                cu_seqlens=cu_seqlens,
                token_positions=token_positions,
                batch_max_seq_len=batch_max_seq_len,
                mode=mode,
                kv_cache=layer_cache,
                cache_slots=cache_slots,
            )

            moe_stats.append(stats)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        aux_loss = t.stack([stat.aux_loss for stat in moe_stats]).mean()

        return ModelOutput(
            logits=logits,
            moe_stats=moe_stats,
            aux_loss=aux_loss
        )
