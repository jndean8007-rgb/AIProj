def allocation_helper(
        cache_manager,
        cont_batch_sched
):

    free_slots = cache_manager.free_slots
    capacity = min(len(free_slots), cont_batch_sched.max_requests - len(cont_batch_sched.active_requests))
    candidate_ids = cont_batch_sched.admission_candidates[:capacity, 0]

    for candidate_id, free_slot in zip(candidate_ids, free_slots):
        cont_batch_sched.admit_request(candidate_id, free_slot)

self.continuous_batch_scheduler = continuous_batch_scheduler

def generate(
        self,
        output_handler: OutputHandler,
        # NO CHANGE THIS continuous_batch_scheduler: ContinuousBatchScheduler,
        max_new_tokens=100,
):
    allocation_helper(
        self.cache_manager,
        self.continuous_batch_scheduler
    )

   ''' for idx in range(min(len(self.cache_manager.free_slots)), )
    for cache_slot in self.cache_manager.free_slots:
'''

    prefill_batch = prefill_batch.to(self.model.device, non_blocking=True)

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
        if decode_batch is not None:
            decode_batch = decode_batch.to(self.model.device, non_blocking=True)

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

            if next_decode_batch is not None:
                next_decode_batch = next_decode_batch.to(self.model.device, non_blocking=True)

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