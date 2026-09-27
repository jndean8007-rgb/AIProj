# LLM System Handoff

Maintained by Claude (see CLAUDE.md). Records architectural decisions and their status.

**Statuses:** Proposed · Agreed · Implemented · Superseded

## Current state

A from-scratch PyTorch + Triton MoE decoder LM (`src/torch_llm`) with a working
training path (packed varlen batches, custom flash-attention and grouped-MoE kernels,
Muon) and a working inference runtime (paged KV cache, continuous batching,
split-K paged decode kernel) as of `930c75a`. Active work (`a2bf43d`, `379a99a`):
FP8 KV-cache quantization, steps 1–2 of the plan in `todo` begun; the tree is
mid-change: the FP8 write path works and is tested; FP8 decode is not implemented yet and is refused with an error.
Roadmap: see D9 / `docs/GOAL_ARCHITECTURE.md` §9. Next: Phase 0a (finish D8),
then Phase 0b (extensibility refactor, D10).

## Decisions

| ID | Decision | Status | Date | Where / notes |
|----|----------|--------|------|---------------|
| D1 | Packed varlen activations everywhere: `[T, ...]` tokens + `cu_seqlens [B+1]` + positions; no padding. | Implemented | pre-2026-09-26 | `model/model.py`, `model/attention.py`, `kernals/flash_attention.py` |
| D2 | GQA attention (`num_q_heads % num_kv_heads == 0`), RoPE on Q/K before caching. | Implemented | pre-2026-09-26 | `model/attention.py`, `model/rope.py` |
| D3 | MoE FFN: top-k router with aux-loss-free bias balancing (`router_beta`, `router_bias_lr`), grouped SwiGLU Triton kernels with custom autograd. | Implemented | pre-2026-09-26 | `model/moe/`, `kernals/moe/` |
| D4 | Muon optimizer with Triton Newton–Schulz / Frobenius kernels. | Implemented | pre-2026-09-26 | `training/optim/muon.py`, `kernals/muon/` |
| D5 | Paged KV cache: per-layer `PagedKVCache` storage `[num_blocks, block_size, H_kv, D]`; allocation, block tables, and lengths owned by a single `KVCacheManager` (CPU-authoritative, device mirror). Storage validates only; bounds/ownership resolved once per batch by the manager. | Implemented | pre-2026-09-26 | `inference/paged_kv_cache.py`, `inference/cache_manager.py` |
| D6 | Continuous batching: `submit`/`step`/`generate`, FIFO admission reserving full lifetime cache capacity up front; prefill and decode are distinct batch types. No preemption/eviction/chunked prefill. | Implemented | pre-2026-09-26 | `inference/runtime.py`, `continuous_batch_scheduler.py`, `inference/README.md` |
| D7 | Decode attention: split-K (flash-decoding) Triton kernel reading paged K/V via the full block table indexed by cache slot. | Implemented | pre-2026-09-26 | `kernals/decode_attention.py` |
| D8 | KV-cache quantization: FP8 e4m3 storage, FP32 scale per (token, KV head) (`scale_granularity="token_head"`), scales stored `[num_blocks, block_size, H_kv]`. Prefill attends over unquantized K/V; only cache writes are quantized. Decode applies scales inside the paged kernel (K scale on the per-token score, V scale on the per-token probability). | Agreed (in progress) | 2026-09-25 | Phase 0a. Steps 1–3 (scale storage, quantized append kernel, verification vs reference) done 2026-09-26 in the working tree: `inference/paged_kv_cache.py`, `kernals/kv_cache/quantized_append.py`; tested by `tests/test_kv_cache_quantization.py`. Next: step 4, decode applies scales (currently guarded by `NotImplementedError` in `kernals/decode_attention.py`). See D11, D12. |
| D9 | Goal architecture (G1–G14): KDA/global 3:1 hybrid (global = MLA → DSA → CSA/HCA, gated, NoPE), pluggable residual (Block AttnRes default / mHC), LatentMoE + shared + hash-early, Engram, shared-weight MTP, Muon family, DeviceMesh FSDP2+EP+CP, unified state manager, OpenAI-compatible serving, agent harness + agentic RAG, roadmap Phases 0–8. | Agreed | 2026-09-26 | `docs/GOAL_ARCHITECTURE.md` §5–§9. Supersedes the informal roadmap in `todo`. |
| D10 | Extensibility contract (G15) + hardware-independent target (G16): code against `SequenceMixer`/`FeedForward`/`Residual`/`TokenMemory`/`OutputHead` protocols; `BatchMeta`; registries + `LayerSpec` config; engine-owned state via `StateSpec` with reserve/commit/truncate/snapshot/restore/free; open `AuxOutputs`; kernel dispatch with references; sharding as policy; versioned configs; generic contract tests. Dev hardware never shapes the architecture. | Agreed | 2026-09-26 | `docs/GOAL_ARCHITECTURE.md` §3–§4. Built in Phase 0b (behavior-preserving refactor). |
| D11 | Unquantized cache stays supported (restores the cache-dtype decision lost in the `2d14592` merge, previously numbered D9): `kv_cache_dtype` in {bf16, fp16, fp32} means plain storage with `k_scales`/`v_scales = None`; `float8_e4m3fn` means quantized storage with scales. Append dispatches on this (plain indexed store, or the Triton quantize kernel; FP8 is CUDA sm89+ only, no CPU FP8 path). The PyTorch reference quantizer lives in `tests/test_kv_cache_quantization.py` as the test oracle, not in production. Decode takes a constexpr `IS_QUANTIZED`. Reason: BF16 is the reference for FP8 error tests, and FP8 casts need sm89+ GPUs. Default `kv_cache_dtype` is bf16; FP8 is opt-in. | Implemented (append side) | 2026-09-26 | 2026-09-26, working tree (uncommitted): `KVCacheConfig` default bf16 + dtype check (`inference/kvcache_config.py`); `PagedKVCache.quantized` flag, scales `None` when unquantized, append dispatch (`inference/paged_kv_cache.py`). Decode-side `IS_QUANTIZED` comes with D8 step 4. Contract tests: `tests/test_kv_cache_quantization.py`. |
| D12 | `scale_granularity` is kept as a config field but only `"token_head"` is valid; the config rejects anything else (restores the lost D10). Reason: per-token scales are computed once at write time and never need rewriting as the cache grows. A second mode (e.g. per-channel K scales for INT4) gets added only when it is actually built. | Implemented | 2026-09-26 | 2026-09-26, working tree (uncommitted): `KVCacheConfig.__post_init__` rejects anything but `"token_head"`. |

## Open questions

- D8 append side: all 57 tests in `tests/test_kv_cache_quantization.py` pass on the user's GPU (2026-09-26). Not yet re-run after these fixes: `test_flash_attention`, `test_norm`, `test_muon_etc`, `test_overfit_tiny` (listed as failing in pytest's cache from an earlier run).
- Pre-existing, unrelated: `test_norm.py` passes 3D input to a 2D-only RMSNorm; `test_inference.py::test_input_output_and_tokenizer_list_contract` expects `"Stream 10 -> 4"` lines but gets `"45"`. (The `muon_kernel_wrapper.py` import is fixed in the working tree.)
