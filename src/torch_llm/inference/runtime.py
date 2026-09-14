import torch as t

from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
from torch_llm.inference.batching import PrefillBatch, build_decode_batch
from torch_llm.inference.kv_cache import KVCache
from torch_llm.inference.sampling import sample
from torch_llm.model.model_config import ModelConfig


class InferenceRuntime:
    def __init__(self, model, model_config: ModelConfig, tokenizer: BPETokenizer):
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer

    def infer(self, prefill_batch: PrefillBatch, max_new_tokens=100):
        kv_caches = [KVCache(
            len(prefill_batch.cu_seqlens) - 1,
            100,
            self.model_config.num_kv_heads,
            self.model_config.head_dim,
            t.float32,
            self.model.device
        ) for _ in range(self.model_config.num_layers)]

        model_output = self.model(
            prefill_batch.token_ids,
            prefill_batch.cu_seqlens,
            prefill_batch.token_positions,
            prefill_batch.batch_max_seq_len,
            mode = 'prefill',
            kv_caches = kv_caches
        )

        next_tokens = sample(model_output, self.tokenizer)

        cache_indices = t.arange(0, len(prefill_batch.cu_seqlens) - 1, dtype=t.long)

       decode_batch = build_decode_batch(

       )
