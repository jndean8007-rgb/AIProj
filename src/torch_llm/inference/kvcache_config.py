from dataclasses import dataclass
import torch as t

@dataclass
class KVCacheConfig:
    num_blocks: int
    block_size: int
    max_cache_slots: int
    cache_max_seq_len: int
    kv_cache_dtype: t.dtype
