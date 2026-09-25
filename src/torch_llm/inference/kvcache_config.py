from dataclasses import dataclass

import torch as t


@dataclass
class KVCacheConfig:
    num_blocks: int = 4096
    block_size: int = 16
    max_cache_slots: int = 64
    cache_max_seq_len: int = 2048
    kv_cache_dtype: t.dtype = t.bfloat16

    def __post_init__(self):
        for name in ("num_blocks", "block_size", "max_cache_slots", "cache_max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two for the paged decode kernel")
        if self.kv_cache_dtype not in (t.float16, t.bfloat16, t.float32):
            raise ValueError("kv_cache_dtype must be float16, bfloat16, or float32")
