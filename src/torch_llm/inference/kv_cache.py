import torch as t


class KVCache:
    """Dense reference cache. The continuous runtime uses PagedKVCache instead."""

    def __init__(self, batch_size, cache_max_seq_len, num_kv_heads, head_dim, dtype, device):
        for value in (batch_size, cache_max_seq_len, num_kv_heads, head_dim):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Cache dimensions must be positive integers")
        if dtype not in (t.float16, t.bfloat16, t.float32):
            raise ValueError("Unsupported KV cache dtype")
        self.batch_size = batch_size
        self.cache_max_seq_len = cache_max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_cache = t.empty((batch_size, cache_max_seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        self.v_cache = t.empty_like(self.k_cache)
        self.seq_lens = t.zeros(batch_size, dtype=t.int32, device=device)

    def _slots(self, cache_slots):
        if cache_slots is None:
            return t.arange(self.batch_size, device=self.k_cache.device)
        slots = t.as_tensor(cache_slots, device=self.k_cache.device)
        if slots.ndim != 1:
            raise ValueError("cache_slots must be a vector of integer indices")
        if slots.numel() == 0:
            return slots.to(t.long)
        if slots.dtype not in (t.int32, t.int64):
            raise ValueError("cache_slots must be a vector of integer indices")
        if not bool(((slots >= 0) & (slots < self.batch_size)).all()) or slots.unique().numel() != slots.numel():
            raise ValueError("cache_slots must be unique, valid cache indices")
        return slots

    def _check_kv(self, new_k, new_v, num_tokens):
        shape = (num_tokens, self.num_kv_heads, self.head_dim)
        if new_k.shape != shape or new_v.shape != shape:
            raise ValueError(f"Expected K/V shape {shape}")
        if new_k.device != self.k_cache.device or new_v.device != self.k_cache.device:
            raise ValueError("K/V must be on the cache device")

    @t.no_grad()
    def append_decode(self, new_k, new_v, cache_slots=None):
        slots = self._slots(cache_slots)
        self._check_kv(new_k, new_v, slots.numel())
        positions = self.seq_lens[slots]
        if not bool((positions < self.cache_max_seq_len).all()):
            raise ValueError("Decode would exceed cache capacity")
        self.k_cache[slots, positions] = new_k.to(self.k_cache.dtype)
        self.v_cache[slots, positions] = new_v.to(self.v_cache.dtype)
        self.seq_lens[slots] += 1

    @t.no_grad()
    def append_prefill(self, new_k, new_v, cu_seqlens, cache_slots=None):
        slots = self._slots(cache_slots)
        self._check_kv(new_k, new_v, new_k.shape[0])
        if cu_seqlens.device != self.k_cache.device or cu_seqlens.dtype not in (t.int32, t.int64):
            raise ValueError("cu_seqlens must be integer metadata on the cache device")
        if cu_seqlens.shape != (slots.numel() + 1,):
            raise ValueError("cu_seqlens must contain one boundary per request plus the end")
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != new_k.shape[0] or not bool((lengths >= 0).all()):
            raise ValueError("Invalid cumulative sequence lengths")
        if not bool((self.seq_lens[slots] + lengths <= self.cache_max_seq_len).all()):
            raise ValueError("Prefill would exceed cache capacity")
        token_slots = t.repeat_interleave(slots, lengths, output_size=new_k.shape[0])
        starts = t.repeat_interleave(cu_seqlens[:-1], lengths, output_size=new_k.shape[0])
        positions = self.seq_lens[token_slots] + t.arange(new_k.shape[0], device=self.k_cache.device) - starts
        self.k_cache[token_slots, positions] = new_k.to(self.k_cache.dtype)
        self.v_cache[token_slots, positions] = new_v.to(self.v_cache.dtype)
        self.seq_lens[slots] += lengths

    def clear(self, cache_slots=None):
        # Stale storage is unreachable once lengths are reset; no large zero-fill.
        self.seq_lens[self._slots(cache_slots)] = 0
