import torch as t

from inference.cache_manager import initialize_cache, CacheContainer
from inference.kvcache_config import KVCacheConfig
from torch_llm.data_pipeline.bpe_tokenizer import BPETokenizer
from torch_llm.inference.batching import PrefillBatch, build_decode_batch, DecodeBatch
from torch_llm.inference.output_handler import OutputHandler
from torch_llm.inference.sampling import sample
from torch_llm.model.model_config import ModelConfig
from typing import Literal


class InferenceRuntime:
    def __init__(
        self,
        model,
        model_config: ModelConfig,
        tokenizer: BPETokenizer,
        kvcache_config: KVCacheConfig,
    ):
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.kvcache_config = kvcache_config

        # Cache lifetime belongs to the runtime
        self.cache_manager, self.paged_kv_caches = initialize_cache(
            kvcache_config=self.kvcache_config,
            model_config=self.model_config,
            device=self.model.device,
        )

        self.next_request_id = 0

    def generate(
        self,
        prefill_batch: PrefillBatch,
        output_handler: OutputHandler,
        max_new_tokens=100,
    ):
        batch_size = len(prefill_batch.cu_seqlens) - 1

        active_stream_indices = list(range(batch_size))
        generated_tokens = [[] for _ in range(batch_size)]

        # One persistent request ID per sequence.
        active_request_ids = list(
            range(
                self.next_request_id,
                self.next_request_id + batch_size,
            )
        )
        self.next_request_id += batch_size

        initial_seq_lens = (
            prefill_batch.cu_seqlens[1:]
            - prefill_batch.cu_seqlens[:-1]
        ).tolist()

        slots = [
            self.cache_manager.allocate_request(
                request_id,
                int(seq_len),
            )
            for request_id, seq_len
            in zip(active_request_ids, initial_seq_lens)
        ]

        active_cache_slots = t.tensor(
            slots,
            dtype=t.long,
            device=self.cache_manager.block_table.device,
        )

        try:
            # -------------------------
            # PREFILL
            # -------------------------

            reserve_batch_capacity(
                active_cache_slots,
                prefill_batch,
                self.cache_manager,
            )

            cache_context = cache_location_context(
                active_cache_slots,
                prefill_batch,
                self.cache_manager,
            )

            decode_batch, active_mask = self.generation_step(
                generated_tokens=generated_tokens,
                max_new_tokens=max_new_tokens,
                active_stream_indices=active_stream_indices,
                batch=prefill_batch,
                cache_batch_context=cache_context,
                mode="prefill",
            )

            # Prefill K/V has now actually been written.
            advance_batch(
                active_request_ids,
                prefill_batch,
                self.cache_manager,
            )

            (active_request_ids, active_cache_slots, active_stream_indices) = \
                filter_finished_requests(
                    active_request_ids,
                    active_cache_slots,
                    active_stream_indices,
                    active_mask,
                    self.cache_manager,
                )

            if decode_batch is None:
                return

            output_handler.handle_output(
                decode_batch,
                active_stream_indices,
            )

            # -------------------------
            # DECODE
            # -------------------------

            while decode_batch is not None and active_stream_indices:
                reserve_batch_capacity(
                    active_cache_slots,
                    decode_batch,
                    self.cache_manager,
                )

                cache_context = cache_location_context(
                    active_cache_slots,
                    decode_batch,
                    self.cache_manager,
                )

                next_decode_batch, active_mask = self.generation_step(
                    generated_tokens=generated_tokens,
                    max_new_tokens=max_new_tokens,
                    active_stream_indices=active_stream_indices,
                    batch=decode_batch,
                    cache_batch_context=cache_context,
                    mode="decode",
                )

                # The CURRENT decode_batch has just been written to cache.
                advance_batch(
                    active_request_ids,
                    decode_batch,
                    self.cache_manager,
                )

                (active_request_ids, active_cache_slots, active_stream_indices) = \
                    filter_finished_requests(
                    active_request_ids,
                    active_cache_slots,
                    active_stream_indices,
                    active_mask,
                    self.cache_manager,
                )

                decode_batch = next_decode_batch

                if decode_batch is not None:
                    output_handler.handle_output(
                        decode_batch,
                        active_stream_indices,
                    )

        finally:
            # Only clean up requests owned by this generate() call.
            for request_id in active_request_ids:
                if request_id in self.cache_manager.request_to_slot:
                    self.cache_manager.free_request(request_id)

    def generation_step(
        self,
        generated_tokens,
        max_new_tokens,
        active_stream_indices,
        batch,
        cache_batch_context,
        mode: Literal["prefill", "decode"] = "prefill",
    ) -> tuple[DecodeBatch | None, list[bool]]:

        last_indices = batch.cu_seqlens[1:] - 1

        model_output = self.model(
            batch.token_ids,
            batch.cu_seqlens,
            batch.token_positions,
            batch.batch_max_seq_len,
            paged_kv_caches=self.paged_kv_caches,
            cache_batch_context=cache_batch_context,
            mode=mode,
        )

        next_tokens = sample(
            model_output.logits[last_indices],
            self.tokenizer,
        )

        # One host transfer for runtime control logic.
        next_token_ids = next_tokens.tolist()

        active_mask = []

        for batch_idx, stream_idx in enumerate(active_stream_indices):
            token = next_token_ids[batch_idx]

            generated_tokens[stream_idx].append(token)

            finished = (
                len(generated_tokens[stream_idx]) >= max_new_tokens
                or token == self.tokenizer.eos_token_id
            )

            active_mask.append(not finished)

        next_batch = build_decode_batch(
            current_batch=batch,
            next_tokens=next_tokens,
            active_mask=active_mask,
        )

        return next_batch, active_mask

def cache_location_context(
    active_cache_slots,
    batch,
    cache_manager,
):
    sequence_lengths = (
        batch.cu_seqlens[1:]
        - batch.cu_seqlens[:-1]
    )

    token_slots = t.repeat_interleave(
        active_cache_slots,
        sequence_lengths,
        output_size=batch.token_positions.numel(),
    )

    physical_blocks, block_offsets = (
        cache_manager.get_physical_location(
            token_slots,
            batch.token_positions,
        )
    )

    context_lens = (
        cache_manager.seq_lens[active_cache_slots]
        + sequence_lengths
    )

    return CacheContainer(
        active_cache_slots,
        physical_blocks,
        block_offsets,
        context_lens,
        cache_manager.block_table,
    )

def reserve_batch_capacity(
    active_cache_slots,
    batch,
    cache_manager,
):
    sequence_lengths = (
        batch.cu_seqlens[1:]
        - batch.cu_seqlens[:-1]
    )

    required_seq_lens = (
        cache_manager.seq_lens[active_cache_slots]
        + sequence_lengths
    )

    slots = active_cache_slots.tolist()
    required_seq_lens = required_seq_lens.tolist()

    for slot, required_seq_len in zip(
        slots,
        required_seq_lens,
    ):
        cache_manager.reserve_capacity(
            int(slot),
            int(required_seq_len),
        )

def advance_batch(
    request_ids,
    batch,
    cache_manager,
):
    sequence_lengths = (
        batch.cu_seqlens[1:]
        - batch.cu_seqlens[:-1]
    ).tolist()

    for request_id, num_tokens in zip(
        request_ids,
        sequence_lengths,
    ):
        cache_manager.advance(
            request_id,
            int(num_tokens),
        )

def filter_finished_requests(
    request_ids,
    cache_slots,
    stream_indices,
    active_mask,
    cache_manager,
):
    next_request_ids = []
    next_stream_indices = []

    for request_id, stream_idx, keep in zip(
        request_ids,
        stream_indices,
        active_mask,
    ):
        if keep:
            next_request_ids.append(request_id)
            next_stream_indices.append(stream_idx)
        else:
            cache_manager.free_request(request_id)

    mask = t.tensor(
        active_mask,
        dtype=t.bool,
        device=cache_slots.device,
    )

    next_cache_slots = cache_slots[mask]

    return (
        next_request_ids,
        next_cache_slots,
        next_stream_indices,
    )