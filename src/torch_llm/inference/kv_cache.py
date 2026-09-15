import torch as t

#need to implement it such that certain batches are removed upon removing from active cache indices

class KVCache:
    def __init__(self,
                 batch_size,
                 cache_max_seq_len,
                 num_kv_heads,
                 head_dim,
                 dtype,
                 device
                 ):
        self.batch_size = batch_size
        self.cache_max_seq_len = cache_max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        #some notion of current sequence lengths
        self.k_cache = t.empty((batch_size, cache_max_seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        self.v_cache = t.empty((batch_size, cache_max_seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        self.seq_lens = t.zeros((batch_size,), dtype=t.int32, device=device)


    def append_decode(self, new_k, new_v, cache_slots: list[int] | None = None):

        assert new_k.shape == new_v.shape
        assert new_k.shape == (self.batch_size, self.num_kv_heads, self.head_dim)
        assert t.all(self.seq_lens < self.cache_max_seq_len).item()

        if cache_slots is None:
            cache_slots = t.arange(self.batch_size, device=self.k_cache.device)

        self.k_cache[cache_slots, self.seq_lens] = new_k
        self.v_cache[cache_slots, self.seq_lens] = new_v

        self.seq_lens += 1

    def append_prefill(self, new_k, new_v, cu_seqlens): #packed token representation T, HKV, D where T = sum of sequence lengths over batches
        assert new_k.shape == new_v.shape
        new_seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        assert cu_seqlens.shape == (self.batch_size + 1,)
        assert cu_seqlens[-1].item() == new_k.shape[0]

        assert t.all(
            self.seq_lens + new_seq_lens <= self.cache_max_seq_len
        ).item()

        batch_ids = t.repeat_interleave(
            t.arange(self.batch_size, device=self.k_cache.device),
            new_seq_lens,
        )

        packed_offsets = t.arange(
            new_k.shape[0],
            device=self.k_cache.device,
        )
        packed_starts = t.repeat_interleave(
            cu_seqlens[:-1],
            new_seq_lens,
        )

        local_positions = packed_offsets - packed_starts

        new_cache_positions = (
            self.seq_lens[batch_ids] + local_positions
        )

        self.k_cache[batch_ids, new_cache_positions] = new_k
        self.v_cache[batch_ids, new_cache_positions] = new_v

        self.seq_lens += new_seq_lens




