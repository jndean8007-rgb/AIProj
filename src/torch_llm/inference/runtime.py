import torch as t

from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
from torch_llm.inference.batching import PrefillBatch, build_decode_batch, DecodeBatch
from torch_llm.inference.kv_cache import KVCache
from torch_llm.inference.output_handler import OutputHandler
from torch_llm.inference.sampling import sample
from torch_llm.model.model_config import ModelConfig
from typing import Literal


class InferenceRuntime:
    def __init__(self, model, model_config: ModelConfig, tokenizer: BPETokenizer):
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer

    def generate(self, prefill_batch: PrefillBatch, output_handler: OutputHandler, max_new_tokens=100):
        batch_size = len(prefill_batch.cu_seqlens) - 1

        kv_caches = [KVCache(
            batch_size,
            self.model_config.model_max_seq_len,
            self.model_config.num_kv_heads,
            self.model_config.head_dim,
            t.bfloat16,
            self.model.device
        ) for _ in range(self.model_config.num_layers)]

        active_stream_indices = list(range(batch_size))


        #CACHE METDATA CONSTRUCTION
        active_cache_indices = list(range(batch_size))


        generated_tokens = [[] for _ in range(batch_size)] # modified in place by generation processor

        #prefill step

        decode_batch = self.generation_step(
            generated_tokens,
            max_new_tokens,
            active_stream_indices,
            prefill_batch,
            kv_caches,
            active_cache_indices,
            mode = 'prefill',
        )

        if not decode_batch:
            return

        output_handler.handle_output(
            decode_batch,
            active_stream_indices,
        )


        #decode loop
        
        while True:
            decode_batch = self.generation_step(
                generated_tokens,
                max_new_tokens,
                active_stream_indices,
                decode_batch,
                kv_caches,
                active_cache_indices,
                mode = 'decode',
            )

            output_handler.handle_output(
                decode_batch,
                active_stream_indices,
            )

            if not active_stream_indices or not decode_batch:
                break

            
    def generation_step(
            self,
            generated_tokens,
            max_new_tokens,
            active_stream_indices,
            batch,
            kv_caches,
            active_cache_indices, # will be changing the active cache indices data blah blah blah
            mode: Literal['prefill', 'decode'] = 'prefill',
    ) -> DecodeBatch | None:

        last_indices = batch.cu_seqlens[1:] - 1

        model_output = self.model(
            batch.token_ids,
            batch.cu_seqlens,
            batch.token_positions,
            batch.batch_max_seq_len,
            kv_caches=kv_caches, #fix to be kache metadata
            cache_slots=active_cache_indices,
            mode = mode
        )

        next_tokens = sample(model_output.logits[last_indices], self.tokenizer)

        active_mask = []
        next_active_stream_indices = []
        next_cache_indices = []


        for batch_idx, stream_idx in enumerate(active_stream_indices):
            token = next_tokens[batch_idx]

            generated_tokens[stream_idx].append(token)

            finished = (
                    len(generated_tokens[stream_idx]) >= max_new_tokens
                    or token == self.tokenizer.eos_token_id
            )

            active_mask.append(not finished)

            if not finished:
                next_active_stream_indices.append(stream_idx)
                next_cache_indices.append(stream_idx)

        active_stream_indices[:] = next_active_stream_indices
        active_cache_indices[:] = next_cache_indices


        return build_decode_batch(
            current_batch = batch,
            next_tokens = next_tokens,
            active_mask=active_mask,
        )
