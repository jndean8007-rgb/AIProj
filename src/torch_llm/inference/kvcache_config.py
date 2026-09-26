from dataclasses import dataclass

import torch as t


@dataclass
class KVCacheConfig:
    num_blocks: int = 4096
    block_size: int = 16
    max_cache_slots: int = 64
    cache_max_seq_len: int = 2048

    kv_cache_dtype: t.dtype = t.float8_e4m3fn
    kv_scale_dtype: t.dtype = t.float32
    scale_granularity: str | None = None
