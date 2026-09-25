# Inference runtime

The runtime owns request scheduling, cache allocation, and per-request generation
state. It runs the model in evaluation mode with autograd disabled. All thirteen
Python modules in this directory were reviewed together with the model/attention
cache consumers, tokenizer, CLI, and relevant tests.

## Public usage and migration

```python
from torch_llm.inference.runtime import InferenceRuntime
from torch_llm.inference.output_handler import OutputHandler

runtime = InferenceRuntime(model, model_config, tokenizer, kvcache_config)
request_id = runtime.submit("A prompt", max_new_tokens=100)
completed = runtime.generate(OutputHandler(tokenizer))
for request in completed:
    print(request.request_id, request.finish_reason)
    print(tokenizer.decode(request.generated_tokens))
```

- `submit()` returns a persistent request ID. Token limits belong to each request;
  `generate()` no longer takes a packed prompt batch or a global token limit.
- `step()` runs pending prefills and one step for requests already decoding. It
  returns the request states that completed in that iteration. Submit new work
  between calls or from an output callback on the owning thread.
- `generate()` drains pending work and returns completed request states in
  completion order. For a long-lived server, consume `step()` results incrementally.
- A custom output sink implements
  `handle_output(token_ids: list[int], request_ids: list[int])`. It receives every
  sampled token, including EOS and the final token at a length limit. The default
  handler prints token diagnostics; reconstruct full text from the token history,
  since individual byte-level BPE tokens may not be complete Unicode characters.
- The tokenizer interface requires `encode(text)` and `eos_token_id`; an EOS ID of
  `None` disables EOS stopping. The console output handler also needs `decode(ids)`.
- Greedy sampling remains the default. Pass a callable as `sampler=`, for example
  `functools.partial(sample, temperature=0.8, top_k=50, top_p=0.95)`. Sampling
  parameters currently apply to the runtime, rather than individual requests.
- `InferenceInputProcessor.prepare(prompts, device="cpu")` remains available for
  direct packed model forwards. It is not the runtime submission interface.
- Mutate scheduling/cache state through its methods. Do not alter queued prompt
  tokens, budgets, cache tensors, or scheduler dictionaries directly after submission.

## Changes, affected behavior, and reasons

| File | What changed | Affected behavior and reason |
| --- | --- | --- |
| `__init__.py` | Added the minimal `InferenceTokenizer` protocol. | Custom tokenizers can satisfy the runtime interface without inheriting from the BPE implementation. |
| `runtime.py` | Replaced duplicate prefill/decode/admission paths with `submit`, `step`, and queue-draining `generate`; kept request state separate from batch rows; added sampler injection and error cleanup. | Fixes replacement IDs, repeated prefilling of active requests, stale cache contexts, omitted final output tokens, missing per-request limits, and leaked active state. Evaluation mode also prevents inference from updating MoE router state. |
| `request_state.py` | Added generated tokens, default unfinished state, finish reasons, prompt/budget validation, and prompt-list copying. | Stopping and results now follow the request across compaction and admission. A zero token budget completes without a model forward. |
| `continuous_batch_scheduler.py` | Enforced capacity and unique pending IDs; returned admitted/completed states; used bounded `islice` for FIFO candidates; removed dead code. | Admission is callable and capacity-aware, and candidate selection no longer copies the entire waiting queue. |
| `batching.py` | Consolidated packed tensor handling; standardized token IDs to int64 and positions/boundaries to int32; constructed packed prompt tensors once; supported iterable requests and empty decode merges. | Fixes promoted int64 cumulative lengths and invalid empty concatenations. Compaction preserves positions and avoids a separate CUDA scalar read for CPU stopping masks. Prefill and decode remain distinct types. |
| `cache_manager.py` | Added CPU ownership/length metadata, explicit capacity errors, batch reservation checks and length commits, safe clear/free, scalar-slot append support, and one context-construction path. | Fixes freed block IDs becoming `-1` through a cleared tensor view, dictionary mutation during clear, inconsistent full/selected block tables, missing decode history, and cache overflow. Freeing no longer reads GPU metadata back to the CPU. |
| `paged_kv_cache.py` | Validated storage and write metadata; explicitly converted K/V to the configured dtype; disabled gradient recording for cache writes. | Prevents shape/device mismatches, mixed-dtype indexed-assignment errors, and cache storage retaining autograd graphs. Bounds and ownership remain the manager's responsibility. |
| `kv_cache.py` | Repaired selected-slot decode writes and length updates; added selected-slot prefill/reset, capacity checks, and gradient-free writes. | Dense reference caches now work after requests finish or batch rows change; inactive slots are not advanced or overwritten. The runtime continues to use paged caches. |
| `kvcache_config.py` | Validated positive dimensions, supported floating dtypes, and power-of-two page sizes. | Invalid configurations fail before allocation or a paged decode kernel launch. |
| `input_processor.py` | Converted encoded prompts to the request-based prefill contract and exposed the destination device. | Fixes passing tensors to a builder that expected request objects, and its missing device argument. |
| `output_handler.py` | Added the structural `OutputSink` interface; accepted CPU token IDs with persistent request IDs; passed token lists to decoding. | Output is decoupled from compacted decode batches and does not require per-token GPU readbacks. |
| `sampling.py` | Kept greedy defaults and added validated temperature, top-k, top-p, and optional generator support without mutating logits. | Sampling policy can be configured or replaced without editing runtime scheduling. The old tokenizer argument remains accepted but unused. |
| `generation_setup.py` | Loaded checkpoints onto CPU with `weights_only=True`, accepted training checkpoints or plain state dictionaries, and returned an eval-mode model. | Avoids restoring saved CUDA/optimizer tensors onto GPU and makes setup consistent with inference execution. |

All package imports in this folder now use the `torch_llm` namespace. Obsolete
commented implementations were removed.

Two production integration files outside the folder also changed:

- `../main.py`: submits the whole input string as one request and calls the new
  runtime API. Removed unused eager training imports that caused inference startup
  to fail on an unrelated training-kernel import.
- `../data_pipeline/bpe_tokenizer.py`: passes the token list directly to the
  underlying decoder, fixing the nested-list bug and matching its declared API.

`tests/test_inference.py` adds lifecycle, cache, batching, sampling, and input/output
regressions. The existing attention, decode-attention, and transformer inference
tests were migrated from obsolete dense-cache call signatures to paged caches,
preserving numerical reference checks and adding ragged batches and mixed precision.

Checkpoint loading follows the
[PyTorch loading guidance](https://docs.pytorch.org/docs/stable/generated/torch.load).

## Capacity and scaling boundaries

Admission is FIFO and reserves each request's bounded lifetime cache capacity:
`min(prompt_length + max_new_tokens - 1, model_context_limit, cache_context_limit)`.
The final sampled token is returned without being fed back through the model.
Thus a prompt that fills the input context can still produce one token; further
generation stops with `context_length`. EOS and requested-length stopping are
reported as `eos` and `length`. Requests that cannot fit the entire page pool are
rejected before admission.

Reserving capacity up front guarantees admitted requests have space to finish.
It can reduce concurrency when budgets are much larger than actual EOS lengths,
and FIFO may leave space idle behind a large waiting request. Incremental page
growth remains available through the manager, but admitting against incremental
capacity would require an explicit eviction/preemption or future-capacity policy.

The runtime uses one owning thread and device/stream. It does not implement
concurrent producers, chunked prefill, mixed prefill/decode kernels, prefix sharing,
eviction, cancellation, speculative decoding, quantized KV caches, or CUDA graphs.
These are architectural extensions, not implied by the current interfaces.
The existing decode kernel still builds split metadata and synchronizes for a
launch-size scalar; that kernel is outside this folder and was not changed.
End-to-end throughput and peak memory were not benchmarked.

## Validation

On the project's Python 3.14.3 / PyTorch 2.13.0+cu130 environment, with CUDA and
Triton available:

```text
pytest tests/test_inference.py tests/test_decode_attention.py tests/test_attention.py tests/test_transformer_lm.py tests/test_transformer_lm_train_backward.py -q
50 passed
```

Coverage includes CPU/CUDA metadata and cache lifecycle checks; 60 queued requests
under repeated page/slot reuse; new submissions during output callbacks; zero,
per-request length, EOS and context limits; model/sampler/output failures;
noncontiguous persistent cache slots; FP32/FP16/BF16 paged-attention comparisons;
FP16/BF16 cached/full-model comparisons; full recomputation versus runtime output;
unchanged model/router state; checkpoint loading; and training backward.

The wider suite reached 56 passes with `--continue-on-collection-errors`. Its
remaining failures are outside the changed inference path:

- `test_muon_etc.py` and `test_overfit_tiny.py` fail collection because
  `kernals/muon_kernel_wrapper.py` imports the nonexistent top-level `kernals` package.
- `test_norm.py` passes a 3D CPU tensor to an RMSNorm implementation requiring
  a packed 2D input.

Those unrelated training/RMSNorm implementations and tests were left unchanged.
