import torch as t

class PagedKVCache:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device,
        dtype,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.k = t.empty(
            (num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        self.v = t.empty(
            (num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

    def append_kv(self, blocks, offsets, ks, vs):
        self.k[blocks, offsets] = ks
        self.v[blocks, offsets] = vs
