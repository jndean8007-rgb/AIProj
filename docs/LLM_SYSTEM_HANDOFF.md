# LLM System Handoff

Maintained by Claude (see CLAUDE.md). Records architectural decisions and their status.

**Statuses:** Proposed · Agreed · Implemented · Superseded

## Current state

A from-scratch PyTorch + Triton MoE decoder LM (`src/torch_llm`) with a working
training path (packed varlen batches, custom flash-attention and grouped-MoE kernels,
Muon) and a working inference runtime (paged KV cache, continuous batching,
split-K paged decode kernel) as of `930c75a`. Active work (`a2bf43d`, `379a99a`):
FP8 KV-cache quantization, steps 1–2 of the plan in `todo` begun; the tree is
mid-change and the inference path is currently broken (see Open questions).
Roadmap after this: distributed training/inference → speculative decoding →
expert parallelism → experimental architectures.

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
| D9 | Goal architecture and phased roadmap (DeepSeek-V4-style hybrid attention, mHC, MoE v2, MTP, heterogeneous cache manager, OpenAI-compatible serving, tool/RAG agent layer). Individual items G1–G13 become Agreed as accepted. | Proposed | 2026-09-26 | `docs/GOAL_ARCHITECTURE.md` |
| D8 | KV-cache quantization: FP8 e4m3 storage, FP32 scale per (token, KV head) (`scale_granularity="token_head"`), scales stored `[num_blocks, block_size, H_kv]`. Prefill attends over unquantized K/V; only cache writes are quantized. Decode dequantizes inside the paged kernel. | Agreed (in progress) | 2026-09-25 | Plan in `todo`. Storage + append kernel begun: `paged_kv_cache.py`, `kernals/kv_cache/quantized_append.py`. Decode-side dequant not started. |

## Open questions

- **D8 breakages in the current tree** (re-checked 2026-09-26 against uncommitted edits to `cache_manager.py`, `kvcache_config.py`, `quantized_append.py`):
  - Fixed: `KVCacheConfig` dtype check (the whole `__post_init__` was removed, so positive-int and power-of-two `block_size` validation are gone too; that is probably unintended).
  - Fixed: wrapper now reads `k_scales`/`v_scales`.
  - Partly fixed: `initialize_cache` now passes `cache_dtype`/`scale_dtype`, but `PagedKVCache` still requires `scale_granularity` (no default), so construction raises `TypeError`.
  - Still broken: `paged_kv_cache.py` imports `kernals...` instead of `torch_llm.kernals...`.
  - Still broken: the append kernel now takes `cache_dtype` as a runtime arg. Triton cannot pass a `torch.dtype` at runtime, and `tl.cast` needs a constexpr `tl.dtype`, so the dtype has to be a constexpr (for example, map `torch.float8_e4m3fn` to `tl.float8e4nv`). There is still no clamp to ±finfo.max before the FP8 cast, and contiguous `ks`/`vs` are still assumed.
  - Still broken: tests construct `PagedKVCache(..., dtype)` with the old signature.
- Should non-FP8 cache dtypes (bf16/fp16) remain supported as an unquantized path (no scales), or is the cache FP8-only from now on?
- Is `scale_granularity` meant to become a real switch (e.g. per-block/per-head), or fixed at `token_head`?
- Pre-existing, unrelated: `kernals/muon_kernel_wrapper.py` top-level `kernals` import breaks `test_muon_etc.py`/`test_overfit_tiny.py` collection; `test_norm.py` passes 3D input to a 2D-only RMSNorm.
