import torch as t

from kernals.kv_cache.quantized_append import quantize_append_wrapper


class PagedKVCache:
    """Per-layer paged storage; allocation and lengths belong to KVCacheManager."""

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int, device, cache_dtype, scale_dtype, scale_granularity):

        for value in (num_blocks, block_size, num_kv_heads, head_dim):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Cache dimensions must be positive integers")

        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_blocks, block_size, num_kv_heads, head_dim)

        self.k = t.empty(shape, device=device, dtype=cache_dtype)
        self.v = t.empty_like(self.k)

        self.k_scales = t.empty(shape[:3], device=device, dtype=scale_dtype)
        self.v_scales = t.empty_like(self.k_scales)

    @t.no_grad()
    def append_kv(self, blocks, offsets, ks, vs):
        # Bounds/ownership are resolved once per batch by the manager, not once
        # per layer. These metadata checks do not synchronize CUDA.

        expected_shape = (blocks.numel(), *self.k.shape[2:])

        if blocks.ndim != 1 or offsets.shape != blocks.shape or ks.shape != expected_shape or vs.shape != expected_shape:
            raise ValueError("K/V and cache locations must describe the same packed tokens")

        if blocks.dtype not in (t.int32, t.int64) or offsets.dtype not in (t.int32, t.int64):
            raise ValueError("Cache locations must be integer tensors")

        if any(value.device != self.k.device for value in (blocks, offsets, ks, vs)):
            raise ValueError("K/V and cache locations must be on the cache device")

        quantize_append_wrapper(
            self,
            ks,
            vs,
            blocks,
            offsets,
        )
