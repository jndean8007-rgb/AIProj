from collections.abc import Sequence
from dataclasses import dataclass

import torch as t

from torch_llm.inference.paged_kv_cache import PagedKVCache


class CacheCapacityError(RuntimeError):
    """The cache pool cannot satisfy an otherwise valid allocation."""


def initialize_cache(kvcache_config, model_config, device):
    """Allocate shared page metadata and independent K/V storage for every layer."""
    manager = KVCacheManager(
        kvcache_config.num_blocks, kvcache_config.block_size,
        kvcache_config.max_cache_slots, kvcache_config.cache_max_seq_len, device,
    )
    caches = [
        PagedKVCache(
            kvcache_config.num_blocks, kvcache_config.block_size,
            model_config.num_kv_heads, model_config.head_dim, device,
            dtype=kvcache_config.kv_cache_dtype,
        )
        for _ in range(model_config.num_layers)
    ]
    return manager, caches


class KVCacheManager:
    """Own page allocation on the CPU and mirror kernel metadata on the device.

    Capacity is reserved separately from committed sequence length. Mutations must
    run on one owning thread/stream; this allocator does not provide locking.
    """

    def __init__(self, num_blocks, block_size, max_cache_slots, max_seq_len, device):
        for name, value in (
            ("num_blocks", num_blocks), ("block_size", block_size),
            ("max_cache_slots", max_cache_slots), ("max_seq_len", max_seq_len),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.max_blocks_per_request = (max_seq_len + block_size - 1) // block_size
        self.free_blocks = list(range(num_blocks))
        self.free_slots = list(range(max_cache_slots))
        self.request_to_slot: dict[int, int] = {}
        self.block_table = t.full(
            (max_cache_slots, self.max_blocks_per_request), -1,
            dtype=t.int32, device=device,
        )
        self.seq_lens = t.zeros(max_cache_slots, dtype=t.int32, device=device)
        self.slot_num_blocks = [0] * max_cache_slots
        self._slot_requests: list[int | None] = [None] * max_cache_slots
        self._slot_blocks: list[list[int]] = [[] for _ in range(max_cache_slots)]
        self._slot_seq_lens = [0] * max_cache_slots

    def required_blocks(self, seq_len: int) -> int:
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or not 0 <= seq_len <= self.max_seq_len:
            raise ValueError(f"Sequence length must be an integer in [0, {self.max_seq_len}]")
        return (seq_len + self.block_size - 1) // self.block_size

    def _check_slot(self, slot: int):
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < len(self._slot_requests):
            raise ValueError("Invalid cache slot")
        if self._slot_requests[slot] is None:
            raise ValueError(f"Cache slot {slot} is not allocated")

    def allocate_request(self, request_id: int, initial_seq_len: int) -> int:
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            raise ValueError("request_id must be an integer")
        if request_id in self.request_to_slot:
            raise ValueError(f"Request {request_id} already owns a cache slot")
        blocks = self.required_blocks(initial_seq_len)
        if not self.free_slots or blocks > len(self.free_blocks):
            raise CacheCapacityError("Insufficient cache slots or blocks")
        slot = self.free_slots.pop()
        self.request_to_slot[request_id] = slot
        self._slot_requests[slot] = request_id
        try:
            self.reserve_capacity(slot, initial_seq_len)
        except Exception:
            self.free_request(request_id)
            raise
        return slot

    def reserve_capacity(self, slot: int, required_seq_len: int):
        self.reserve_batch_capacity([slot], [required_seq_len])

    def reserve_batch_capacity(self, slots: Sequence[int], required_seq_lens: Sequence[int]):
        """Check the whole batch before taking any blocks from the free pool."""
        if len(slots) != len(required_seq_lens) or len(set(slots)) != len(slots):
            raise ValueError("Capacity reservations require one length per unique slot")
        reservations = []
        for slot, length in zip(slots, required_seq_lens):
            self._check_slot(slot)
            required = self.required_blocks(length)
            reservations.append((slot, max(0, required - self.slot_num_blocks[slot])))
        if sum(count for _, count in reservations) > len(self.free_blocks):
            raise CacheCapacityError("Insufficient cache blocks")
        for slot, count in reservations:
            if count == 0:
                continue
            allocated = self.free_blocks[-count:]
            start = self.slot_num_blocks[slot]
            self.block_table[slot, start:start + count] = t.tensor(
                allocated, dtype=t.int32, device=self.block_table.device,
            )
            del self.free_blocks[-count:]
            self._slot_blocks[slot].extend(allocated)
            self.slot_num_blocks[slot] += count

    def advance(self, request_id: int, num_tokens: int):
        self.advance_batch([request_id], [num_tokens])

    def advance_batch(self, request_ids: Sequence[int], num_tokens: Sequence[int]):
        if len(request_ids) != len(num_tokens) or len(set(request_ids)) != len(request_ids):
            raise ValueError("Advancing requires one token count per unique request")
        slots, lengths = [], []
        for request_id, count in zip(request_ids, num_tokens):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("num_tokens must be a nonnegative integer")
            slot = self.get_slot(request_id)
            length = self._slot_seq_lens[slot] + count
            if length > self.max_seq_len or length > self.slot_num_blocks[slot] * self.block_size:
                raise CacheCapacityError("Sequence exceeds its reserved cache capacity")
            slots.append(slot)
            lengths.append(length)
        if slots:
            self.seq_lens[t.tensor(slots, device=self.seq_lens.device, dtype=t.long)] = t.tensor(
                lengths, device=self.seq_lens.device, dtype=t.int32,
            )
            for slot, length in zip(slots, lengths):
                self._slot_seq_lens[slot] = length

    def get_slot(self, request_id: int) -> int:
        return self.request_to_slot[request_id]

    def get_seq_len(self, request_id: int) -> int:
        return self._slot_seq_lens[self.get_slot(request_id)]

    def _indices(self, values) -> t.Tensor:
        values = t.as_tensor(values, device=self.block_table.device)
        if values.dtype not in (t.int32, t.int64):
            raise ValueError("Cache indices must be int32 or int64")
        return values

    def get_physical_location(self, slots, token_positions: t.Tensor, *, validate=True):
        slots = self._indices(slots)
        token_positions = self._indices(token_positions)
        slots, token_positions = t.broadcast_tensors(slots, token_positions)
        if validate:
            valid = (
                (slots >= 0) & (slots < self.block_table.shape[0])
                & (token_positions >= 0) & (token_positions < self.max_seq_len)
            )
            if not bool(valid.all()):
                raise ValueError("Cache slots or token positions are out of bounds")
        logical_blocks = token_positions // self.block_size
        physical_blocks = self.block_table[slots, logical_blocks]
        if validate and not bool((physical_blocks >= 0).all()):
            raise ValueError("Token position has no reserved cache block")
        return physical_blocks, token_positions % self.block_size

    def free_request(self, request_id: int):
        slot = self.get_slot(request_id)
        # CPU ownership is authoritative: clearing a tensor view must never turn
        # the freed page IDs into -1, and freeing must not read back CUDA memory.
        self.block_table[slot].fill_(-1)
        self.seq_lens[slot] = 0
        self.free_blocks.extend(self._slot_blocks[slot])
        self._slot_blocks[slot] = []
        self.slot_num_blocks[slot] = 0
        self._slot_seq_lens[slot] = 0
        self._slot_requests[slot] = None
        del self.request_to_slot[request_id]
        self.free_slots.append(slot)

    def clear(self):
        for request_id in list(self.request_to_slot):
            self.free_request(request_id)

    def get_block_table(self, slots):
        slots = self._indices(slots)
        for slot in slots.reshape(-1).tolist():
            self._check_slot(slot)
        return self.block_table[slots]

    def append_request(self, request_id, token_positions, ks, vs, paged_cache):
        blocks, offsets = self.get_physical_location(self.get_slot(request_id), token_positions)
        paged_cache.append_kv(blocks, offsets, ks, vs)

    def resolve_batch_locations(self, cache_slots, cu_seqlens, token_positions, *, validate=True):
        if not (cache_slots.device == cu_seqlens.device == token_positions.device == self.block_table.device):
            raise ValueError("Batch metadata and cache must be on the same device")
        if cache_slots.ndim != 1 or cu_seqlens.shape != (cache_slots.numel() + 1,) or token_positions.ndim != 1:
            raise ValueError("Invalid packed batch metadata shapes")
        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if validate and (
            int(cu_seqlens[0]) != 0
            or int(cu_seqlens[-1]) != token_positions.numel()
            or not bool((sequence_lengths > 0).all())
        ):
            raise ValueError("Invalid cumulative sequence lengths")
        token_slots = t.repeat_interleave(
            cache_slots, sequence_lengths, output_size=token_positions.numel(),
        )
        return self.get_physical_location(token_slots, token_positions, validate=validate)

    def create_container(self, cache_slots, cu_seqlens, token_positions, *, validate=True):
        """Describe writes and full attention context before committing this batch.

        validate=False is for runtime-built, already allocated metadata only; it
        avoids device-to-host bounds checks on the generation path.
        """
        physical_blocks, block_offsets = self.resolve_batch_locations(
            cache_slots, cu_seqlens, token_positions, validate=validate,
        )
        context_lengths = self.seq_lens[cache_slots] + cu_seqlens[1:] - cu_seqlens[:-1]
        return CacheContainer(
            cache_slots, physical_blocks, block_offsets, context_lengths,
            self.block_table,
        )


@dataclass
class CacheContainer:
    cache_slots: t.Tensor
    physical_blocks: t.Tensor
    block_offsets: t.Tensor
    context_lengths: t.Tensor
    # Full table, indexed by persistent cache_slots inside the decode kernel.
    block_table: t.Tensor
