from collections.abc import Callable
from typing import Literal

import torch as t

from torch_llm.inference import InferenceTokenizer

from torch_llm.inference.batching import (
    DecodeBatch, PrefillBatch, build_decode_batch, build_prefill_batch, merge_decode_batches,
)
from torch_llm.inference.cache_manager import initialize_cache
from torch_llm.inference.continuous_batch_scheduler import ContinuousBatchScheduler
from torch_llm.inference.kvcache_config import KVCacheConfig
from torch_llm.inference.output_handler import OutputSink
from torch_llm.inference.request_state import RequestState
from torch_llm.inference.sampling import sample
from torch_llm.model.model_config import ModelConfig

class InferenceRuntime:
    """Single-owner continuous batching with stable request IDs.

    submit() queues work, step() performs one scheduling iteration, and generate()
    drains the queue. Requests may be submitted between steps or by output
    callbacks on the owning thread. Concurrent/reentrant execution is unsupported.
    """

    def __init__(
        self,
        model,
        model_config: ModelConfig,
        tokenizer: InferenceTokenizer,
        kvcache_config: KVCacheConfig,
        continuous_batch_scheduler: ContinuousBatchScheduler | None = None,
        *,
        sampler: Callable[[t.Tensor], t.Tensor] = sample,
    ):
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.kvcache_config = kvcache_config
        self.continuous_batch_scheduler = continuous_batch_scheduler or ContinuousBatchScheduler(
            kvcache_config.max_cache_slots,
        )
        if self.continuous_batch_scheduler.active_requests:
            raise ValueError("The runtime requires a scheduler with no already active requests")
        self.device = next(model.parameters()).device
        self.max_context_len = min(kvcache_config.cache_max_seq_len, model_config.model_max_seq_len)
        self.sampler = sampler
        self.eos_token_id = tokenizer.eos_token_id
        self.cache_manager, self.paged_kv_caches = initialize_cache(
            kvcache_config, model_config, self.device,
        )
        self.next_request_id = max(self.continuous_batch_scheduler.waiting_requests, default=-1) + 1
        self._decode_batch: DecodeBatch | None = None
        self._decode_requests: list[RequestState] = []
        self._waiting_capacities: dict[int, int] = {}
        self._stepping = False
        self.model.eval()

    def _required_capacity(self, request: RequestState) -> int:
        prompt_len = len(request.prompt_tokens)
        if prompt_len > self.max_context_len:
            raise ValueError(f"Prompt exceeds the supported context length ({self.max_context_len})")
        if any(token >= self.model_config.vocab_size for token in request.prompt_tokens):
            raise ValueError("Prompt contains token IDs outside the model vocabulary")
        if request.max_new_tokens == 0:
            return 0
        # The final sampled token is returned without being fed back to the model.
        capacity = min(prompt_len + request.max_new_tokens - 1, self.max_context_len)
        if self.cache_manager.required_blocks(capacity) > self.cache_manager.num_blocks:
            raise ValueError("Request cannot fit in the entire KV cache pool")
        return capacity

    def submit(self, prompt: str, max_new_tokens: int = 100, continuous_batch_scheduler=None) -> int:
        if continuous_batch_scheduler is not None and continuous_batch_scheduler is not self.continuous_batch_scheduler:
            raise ValueError("Submit requests to the scheduler owned by this runtime")
        scheduler = self.continuous_batch_scheduler
        while self.next_request_id in scheduler.waiting_requests or self.next_request_id in scheduler.active_requests:
            self.next_request_id += 1
        request = RequestState(self.next_request_id, self.tokenizer.encode(prompt), max_new_tokens)
        capacity = self._required_capacity(request)
        scheduler.submit(request)
        self._waiting_capacities[request.request_id] = capacity
        self.next_request_id += 1
        return request.request_id

    def _admit_waiting(self) -> tuple[list[RequestState], list[RequestState]]:
        scheduler = self.continuous_batch_scheduler
        admitted, completed = [], []
        for request_id, request in scheduler.admission_candidates():
            if request_id not in self._waiting_capacities:
                self._waiting_capacities[request_id] = self._required_capacity(request)
            capacity = self._waiting_capacities[request_id]
            if request.max_new_tokens == 0:
                scheduler.admit_request(request_id)
                request.finish_reason = "length"
                completed.append(scheduler.remove(request_id))
                del self._waiting_capacities[request_id]
                continue
            # Reserve the bounded lifetime capacity before admission. This FIFO
            # policy guarantees that active requests can finish without eviction;
            # it is conservative when requests terminate early at EOS.
            if not self.cache_manager.free_slots or self.cache_manager.required_blocks(capacity) > len(self.cache_manager.free_blocks):
                break
            self.cache_manager.allocate_request(request_id, capacity)
            scheduler.admit_request(request_id)
            del self._waiting_capacities[request_id]
            admitted.append(request)
        return admitted, completed

    def _run_batch(
        self, batch: PrefillBatch | DecodeBatch, requests: list[RequestState],
        mode: Literal["prefill", "decode"], output_handler: OutputSink | None,
    ) -> tuple[DecodeBatch | None, list[RequestState], list[RequestState]]:
        request_ids = [request.request_id for request in requests]
        slots = t.tensor(
            [self.cache_manager.get_slot(request_id) for request_id in request_ids],
            dtype=t.long, device=self.device,
        )
        context = self.cache_manager.create_container(
            slots, batch.cu_seqlens, batch.token_positions, validate=False,
        )
        output = self.model(
            batch.token_ids, batch.cu_seqlens, batch.token_positions, batch.batch_max_seq_len,
            paged_kv_caches=self.paged_kv_caches, cache_batch_context=context, mode=mode,
        )
        next_tokens = self.sampler(output.logits[batch.cu_seqlens[1:] - 1])
        if next_tokens.shape != (len(requests),) or next_tokens.dtype not in (t.int32, t.int64):
            raise ValueError("The sampler must return one integer token ID per request")
        # One batched transfer supplies stopping decisions, output, and CPU state.
        token_ids = next_tokens.tolist()
        if any(token < 0 or token >= self.model_config.vocab_size for token in token_ids):
            raise ValueError("The sampler returned token IDs outside the model vocabulary")
        self.cache_manager.advance_batch(
            request_ids, [len(request.prompt_tokens) if mode == "prefill" else 1 for request in requests],
        )
        keep_mask, survivors, completed = [], [], []
        for request, token in zip(requests, token_ids):
            request.generated_tokens.append(token)
            if token == self.eos_token_id:
                request.finish_reason = "eos"
            elif len(request.generated_tokens) >= request.max_new_tokens:
                request.finish_reason = "length"
            elif self.cache_manager.get_seq_len(request.request_id) >= self.max_context_len:
                request.finish_reason = "context_length"
            keep = request.finish_reason is None
            keep_mask.append(keep)
            if keep:
                survivors.append(request)
            else:
                self.cache_manager.free_request(request.request_id)
                completed.append(self.continuous_batch_scheduler.remove(request.request_id))
        next_batch = build_decode_batch(batch, next_tokens, keep_mask)
        if output_handler is not None:
            # Emit ALL sampled tokens, including EOS and the last allowed token.
            output_handler.handle_output(token_ids, request_ids)
        return next_batch, survivors, completed

    @t.inference_mode()
    def step(self, output_handler: OutputSink | None = None) -> list[RequestState]:
        """Run new prefills and one decode step; return requests completed now."""
        if self._stepping:
            raise RuntimeError("InferenceRuntime.step() is not reentrant")
        self._stepping = True
        try:
            if self.model.training:
                self.model.eval()
            admitted, completed = self._admit_waiting()
            new_batch, new_requests = None, []
            if admitted:
                new_batch, new_requests, finished = self._run_batch(
                    build_prefill_batch(admitted, self.device), admitted, "prefill", output_handler,
                )
                completed.extend(finished)
            old_batch, old_requests = None, []
            if self._decode_batch is not None:
                old_batch, old_requests, finished = self._run_batch(
                    self._decode_batch, self._decode_requests, "decode", output_handler,
                )
                completed.extend(finished)
            self._decode_batch = merge_decode_batches([old_batch, new_batch])
            self._decode_requests = old_requests + new_requests
            return completed
        except BaseException:
            # Waiting requests remain queued; all admitted work is released even
            # when a model, sampler, or output callback fails.
            for request_id in list(self.cache_manager.request_to_slot):
                self.cache_manager.free_request(request_id)
                request = self.continuous_batch_scheduler.active_requests.get(request_id)
                if request is not None:
                    request.finish_reason = "error"
                    self.continuous_batch_scheduler.remove(request_id)
            self._decode_batch = None
            self._decode_requests = []
            raise
        finally:
            self._stepping = False

    def generate(self, output_handler: OutputSink | None = None) -> list[RequestState]:
        """Drain pending work, returning completed request states in completion order.

        Use step() and consume its results incrementally for a long-lived server,
        so completed output does not accumulate in one generate() result.
        """
        completed = []
        while self.continuous_batch_scheduler.has_pending_requests():
            completed.extend(self.step(output_handler))
        return completed
