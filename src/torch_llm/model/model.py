import torch.nn as nn
import torch as t
from typing_extensions import Literal
from jaxtyping import Shaped

from core.aux_outputs import AuxOutputs
from core.batch_meta import BatchMeta
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
            batch_meta: BatchMeta,
            paged_kv_caches= None,
    ):

        moe_stats = []
        aux_outputs = AuxOutputs()

        x = self.embedding(batch_meta.token_ids)

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if paged_kv_caches is None else paged_kv_caches[layer_idx]

            x, new_aux_outputs, stats = block(
                x=x,
                batch_meta=batch_meta,
                paged_kv_cache=layer_cache,
            )

            moe_stats.append(stats)

            aux_outputs.merged(new_aux_outputs, prefix=f"layers.{layer_idx}")

        x = self.final_norm(x)

        logits = self.lm_head(x)

        aux_loss = t.stack([stat.aux_loss for stat in moe_stats]).mean()

        return ModelOutput(
            logits=logits,
            moe_stats=moe_stats,
            aux_loss=aux_loss
        )

    @property
    def device(self):
        return next(self.parameters()).device