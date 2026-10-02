import torch as t
import torch.nn as nn
from jaxtyping import Shaped

from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta
from torch_llm.model.decoder_block import DecoderBlock
from torch_llm.model.model_config import ModelConfig
from torch_llm.model.outputs import ModelOutputs
from torch_llm.model.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    def __init__(
            self,
            config: ModelConfig,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(config, layer_idx)
                for layer_idx in range(config.num_layers)
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

        # potentially temporary weight tying
        self.lm_head.weight = self.embedding.weight

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(
            self,
            batch_meta: BatchMeta,
            paged_kv_caches=None,
    ):
        aux_outputs = AuxOutputs()

        x = self.embedding(batch_meta.token_ids)

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if paged_kv_caches is None else paged_kv_caches[layer_idx]

            x, new_aux_outputs = block(
                x=x,
                batch_meta=batch_meta,
                paged_kv_cache=layer_cache,
            )

            aux_outputs = aux_outputs.merged(new_aux_outputs, prefix=f"layers.{layer_idx}")

        x = self.final_norm(x)

        logits = self.lm_head(x)

        # aux_loss = t.stack([stat.aux_loss for stat in moe_stats]).mean()

        return ModelOutputs(
            logits=logits,
            aux_outputs=aux_outputs,
        )

    @property
    def device(self):
        return next(self.parameters()).device
