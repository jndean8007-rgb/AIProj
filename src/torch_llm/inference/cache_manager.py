from dataclasses import dataclass

import torch as t

from inference.paged_kv_cache import PagedKVCache


def initialize_cache(
        kvcache_config,
        model_config,
        device,
) -> tuple[KVCacheManager, list[PagedKVCache]]:
    '''
    intialize cache maanger and paged kv caches dependent per layer basis
    :param kvcache_config:
    :param model_config:
    :param device:
    :return:
    '''

    cache_manager = KVCacheManager(
        kvcache_config.num_blocks,
        kvcache_config.block_size,
        kvcache_config.max_cache_slots,
        kvcache_config.cache_max_seq_len,
        device,
    )

    paged_kv_caches = [
        PagedKVCache(
            kvcache_config.num_blocks,
            kvcache_config.block_size,
            model_config.num_kv_heads,
            model_config.head_dim,
            device,
            dtype = kvcache_config.kv_cache_dtype,
        ) for _ in range(model_config.num_layers)
    ]

    return cache_manager, paged_kv_caches


class KVCacheManager:
    def __init__(
            self,
            num_blocks,
            block_size,
            max_cache_slots,
            max_seq_len,
            device,
    ):
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.max_blocks_per_request = (max_seq_len + block_size - 1)// block_size
        self.free_blocks = list(range(num_blocks))
        self.free_slots = list(range(max_cache_slots))

        self.request_to_slot = {}

        self.block_table = t.full(
            (max_cache_slots, self.max_blocks_per_request),
            -1,
            dtype = t.int32,
            device = device
        )

        self.seq_lens = t.full(
            (max_cache_slots,),
            0,
            dtype = t.int32,
            device = device,
        )

        self.slot_num_blocks = [0] * max_cache_slots

    def allocate_request(self, request_id, initial_seq_len) -> int:
        assert request_id not in self.request_to_slot
        assert initial_seq_len <= self.max_seq_len
        assert self.free_slots

        required_blocks = (initial_seq_len + self.block_size - 1) // self.block_size

        assert len(self.free_blocks) >= required_blocks

        slot = self.free_slots.pop()

        self.slot_num_blocks[slot] = required_blocks

        allocated_blocks = [
            self.free_blocks.pop()
            for _ in range(required_blocks)
        ]

        self.request_to_slot[request_id] = slot
        self.seq_lens[slot] = 0

        self.block_table[slot, :required_blocks] = t.tensor(
            allocated_blocks,
            dtype = self.block_table.dtype,
            device=self.block_table.device,
        )

        return slot

    def reserve_capacity(self, request_id, num_tokens):
        assert request_id in self.request_to_slot
        slot = self.request_to_slot[request_id]

        required_seq_len = self.seq_lens[slot] + num_tokens
        assert required_seq_len <= self.max_seq_len

        required_blocks = (required_seq_len + self.block_size - 1) // self.block_size
        new_blocks_required = required_blocks - self.slot_num_blocks[slot]

        if new_blocks_required > 0:
            assert len(self.free_blocks) >= new_blocks_required


            allocated_blocks = [
                self.free_blocks.pop()
                for _ in range(new_blocks_required)
            ]

            self.block_table[slot, self.slot_num_blocks[slot]:required_blocks] = t.tensor(
                allocated_blocks,
                dtype = self.block_table.dtype,
                device=self.block_table.device,
            )

            self.slot_num_blocks[slot] = required_blocks

    def advance(self, request_id, num_tokens):
        assert request_id in self.request_to_slot

        slot = self.request_to_slot[request_id]
        assert num_tokens + self.seq_lens[slot] <= self.slot_num_blocks[slot] * self.block_size

        self.seq_lens[slot] += num_tokens

    def get_slot(self, request_id):
        assert request_id in self.request_to_slot
        return self.request_to_slot[request_id]

    def get_physical_location(self, slots: t.tensor, token_positions: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        assert all(slot not in self.free_slots for slot in slots)
        assert token_positions.device == self.block_table.device

        logical_blocks = token_positions // self.block_size
        physical_blocks = self.block_table[slots, logical_blocks]
        offsets = token_positions % self.block_size

        return physical_blocks, offsets

    def free_request(self, request_id):
        assert request_id in self.request_to_slot
        slot = self.request_to_slot[request_id]
        blocks = self.block_table[slot, :self.slot_num_blocks[slot]]

        self.block_table[slot].fill_(-1)

        self.free_blocks.extend(blocks.tolist())

        self.slot_num_blocks[slot] = 0
        self.seq_lens[slot] = 0
        del self.request_to_slot[request_id]

        self.free_slots.append(slot)

    def clear(self):
        for request_id in self.request_to_slot:
            self.free_request(request_id)

    def get_block_table(self, slots):
        assert all(slot not in self.free_slots for slot in slots)
        return self.block_table[slots]

    def append_request(self, request_id, token_positions, ks, vs, paged_cache):
        assert request_id in self.request_to_slot
        slot = self.get_slot(request_id)
        blocks, offsets = self.get_physical_location(slot, token_positions)
        paged_cache.append_kv(blocks, offsets, ks, vs)

    def resolve_batch_locations(self, cache_slots, cu_seqlens, token_positions):
        assert cache_slots.device == self.block_table.device == token_positions.device == cu_seqlens.device
        assert cu_seqlens.numel() == cache_slots.numel() + 1

        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]

        token_slots = t.repeat_interleave(
            cache_slots,
            sequence_lengths,
            output_size=token_positions.numel(),
        )

        logical_blocks = token_positions // self.block_size


        physical_blocks = self.block_table[token_slots, logical_blocks]
        offsets = token_positions % self.block_size

        return physical_blocks, offsets

    def create_container(self, cache_slots, cu_seqlens, token_positions):
        context_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        physical_blocks, block_offsets = self.resolve_batch_locations(cache_slots, cu_seqlens, token_positions)
        block_table = self.get_block_table(cache_slots)

        return CacheContainer(
            cache_slots,
            physical_blocks,
            block_offsets,
            context_lengths,
            block_table
        )


@dataclass
class CacheContainer:
    cache_slots: t.tensor
    physical_blocks: t.tensor
    block_offsets: t.tensor
    context_lengths: t.tensor
    block_table: t.tensor







