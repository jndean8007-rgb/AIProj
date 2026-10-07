import torch as t
import torch.nn as nn
from jaxtyping import Shaped

from torch_llm.core.aux_outputs import AuxOutputs
from torch_llm.core.batch_meta import BatchMeta
from torch_llm.core.registry import OUTPUT_HEADS, RESIDUALS, FFNS
from torch_llm.model.decoder_block import DecoderBlock
from torch_llm.model.model_config import ModelConfig
from torch_llm.model.outputs import ModelOutputs
import torch_llm.model.components  # noqa: F401  (imports every component so the registries are filled)


class TransformerLM(nn.Module):
    def __init__(
            self,
            config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # Built once with the model; the stream itself is created per forward by residual.init.
        self.residual = RESIDUALS.get(config.residual)(config)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )

        # Owns the final norm and ties its projection to the embedding (replaces final_norm + lm_head).
        # Built after the blocks so parameter order matches the old layout.
        self.head = OUTPUT_HEADS.get(config.output_head)(config, self.embedding)

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

        state = self.embedding(batch_meta.token_ids)

        state = self.residual.init(state)

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None if paged_kv_caches is None else paged_kv_caches[layer_idx]

            state, new_aux_outputs = block(
                state=state,
                residual=self.residual,
                batch_meta=batch_meta,
                paged_kv_cache=layer_cache,
            )

            aux_outputs = aux_outputs.merged(new_aux_outputs, prefix=f"layers.{layer_idx}")

        logits, head_aux_outputs = self.head(self.residual.final(state), batch_meta)
        aux_outputs = aux_outputs.merged(head_aux_outputs, prefix="head")

        return ModelOutputs(
            logits=logits,
            aux_outputs=aux_outputs,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def post_step(self) -> None:
        for layer_idx, block in enumerate(self.blocks):
            block.ffn.post_step()

