from dataclasses import dataclass

import torch as t


@dataclass
class KVCacheConfig:
    num_blocks: int = 4096
    block_size: int = 16
    max_cache_slots: int = 64
    cache_max_seq_len: int = 2048

    kv_cache_dtype: t.dtype = t.bfloat16
    kv_scale_dtype: t.dtype = t.float32
    scale_granularity: str = "token_head"

    def  __post_init__(self):
        if any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in (
                        self.num_blocks,
                        self.block_size,
                        self.max_cache_slots,
                        self.cache_max_seq_len,
                )
        ):
            raise ValueError("All cache dimensions must be positive integers.")

        if not (self.block_size & (self.block_size - 1)) == 0:
            raise ValueError("Block size must be a power of 2.")

        if not self.kv_cache_dtype in [t.float16, t.bfloat16, t.float32, t.float8_e4m3fn]:
            raise ValueError("KVCache dtype must be in: [t.float16, t.bfloat16, t.float32, t.float8_e4m3fn]")

        if not self.kv_scale_dtype == t.float32:
            raise ValueError("KVScale dtype must be t.float32")

        if not self.scale_granularity == "token_head":
            raise ValueError("scale granularity is only supported for token_head")