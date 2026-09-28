"""Paged decode attention with a BF16 vs FP8 KV cache (handoff D8, step 6).

Reports:
  1. KV-cache bytes per token per layer, from the real tensor sizes.
  2. A measured device-memory bandwidth (large copy), as the practical roofline.
  3. Decode-attention wall time per call across batch sizes and context lengths,
     with the KV bytes each call must read and the bandwidth that implies.
  4. A least-squares fit per dtype of   time = overhead + kv_bytes / bandwidth.
     The intercept is the fixed cost per call (launches, the host sync in the
     wrapper); the slope's inverse is the bandwidth the kernel achieves once the
     KV read dominates. FP8 can only speed up the second term.

Run (from the repository root):
  .\\.venv\\Scripts\\python.exe benchmarks/bench_kv_cache_fp8.py
  .\\.venv\\Scripts\\python.exe benchmarks/bench_kv_cache_fp8.py --head-dim 128 --context-lengths 1024 4096 16384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch as t

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_llm.inference.cache_manager import KVCacheManager
from torch_llm.inference.paged_kv_cache import PagedKVCache
from torch_llm.kernals.decode_attention import decode_attention_wrapper

FP8 = t.float8_e4m3fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--num-q-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def time_per_call_ms(fn, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    t.cuda.synchronize()
    start, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    t.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def measure_copy_bandwidth_gbs(nbytes: int = 512 * 2**20) -> float:
    source = t.empty(nbytes, dtype=t.uint8, device="cuda")
    target = t.empty_like(source)
    ms = time_per_call_ms(lambda: target.copy_(source), iterations=20, warmup=5)
    return 2 * nbytes / (ms * 1e-3) / 1e9  # a copy reads and writes every byte


def cache_bytes_per_token(cache: PagedKVCache) -> int:
    """Bytes one token occupies in one layer: K and V vectors for every KV head, plus scales."""
    num_kv_heads, head_dim = cache.k.shape[2:]
    per_vector = head_dim * cache.k.element_size()
    if cache.quantized:
        per_vector += cache.k_scales.element_size()
    return 2 * num_kv_heads * per_vector


def build_decode_batch(batch_size, context_length, args):
    """One manager, a BF16 cache and an FP8 cache holding the same K/V history."""
    block_size = args.block_size
    blocks_per_request = -(-context_length // block_size)
    num_blocks = batch_size * blocks_per_request + 1
    manager = KVCacheManager(num_blocks, block_size, batch_size + 1, context_length + block_size, "cuda")
    caches = {
        dtype: PagedKVCache(num_blocks, block_size, args.num_kv_heads, args.head_dim, "cuda", dtype)
        for dtype in (t.bfloat16, FP8)
    }
    slots = []
    for request_id in range(batch_size):
        slots.append(manager.allocate_request(request_id, context_length))
        k = t.randn(context_length, args.num_kv_heads, args.head_dim, device="cuda", dtype=t.bfloat16)
        v = t.randn_like(k)
        for cache in caches.values():
            manager.append_request(request_id, t.arange(context_length, device="cuda"), k, v, cache)
        manager.advance(request_id, context_length - 1)
    context = manager.create_container(
        t.tensor(slots, device="cuda"),
        t.arange(batch_size + 1, dtype=t.int32, device="cuda"),
        t.full((batch_size,), context_length - 1, device="cuda"),
    )
    return caches, context


def main() -> None:
    args = parse_args()
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"device={t.cuda.get_device_name()} capability={t.cuda.get_device_capability()} torch={t.__version__}")
    print(f"heads: q={args.num_q_heads} kv={args.num_kv_heads} head_dim={args.head_dim} block_size={args.block_size}\n")

    copy_gbs = measure_copy_bandwidth_gbs()
    print(f"measured copy bandwidth (read + write): {copy_gbs:.0f} GB/s\n")

    rows = []
    with t.inference_mode():
        for batch_size in args.batch_sizes:
            for context_length in args.context_lengths:
                caches, context = build_decode_batch(batch_size, context_length, args)
                q = t.randn(batch_size, args.num_q_heads, args.head_dim, device="cuda", dtype=t.bfloat16)
                for dtype, cache in caches.items():
                    kv_bytes = batch_size * context_length * cache_bytes_per_token(cache)
                    ms = time_per_call_ms(lambda: decode_attention_wrapper(q, cache, context), args.iterations, args.warmup)
                    rows.append((dtype, batch_size, context_length, kv_bytes, ms))
                del caches, context, q
                t.cuda.empty_cache()

    names = {t.bfloat16: "bf16", FP8: "fp8"}
    sample = {dtype: PagedKVCache(1, args.block_size, args.num_kv_heads, args.head_dim, "cuda", dtype) for dtype in names}
    bf16_bytes, fp8_bytes = (cache_bytes_per_token(sample[d]) for d in (t.bfloat16, FP8))
    print(f"KV bytes per token per layer: bf16={bf16_bytes}  fp8={fp8_bytes}  (fp8/bf16 = {fp8_bytes / bf16_bytes:.3f})\n")

    print(f"{'dtype':>5} {'batch':>5} {'context':>7} {'KV MB':>8} {'us/call':>9} {'GB/s':>7} {'% copy':>7}")
    for dtype, batch_size, context_length, kv_bytes, ms in rows:
        gbs = kv_bytes / (ms * 1e-3) / 1e9
        print(f"{names[dtype]:>5} {batch_size:>5} {context_length:>7} {kv_bytes / 2**20:>8.1f} {ms * 1e3:>9.1f} {gbs:>7.0f} {100 * gbs / copy_gbs:>6.0f}%")

    print(f"\n{'batch':>5} {'context':>7} {'fp8 speedup':>12}")
    timing = {(d, b, c): ms for d, b, c, _, ms in rows}
    for batch_size in args.batch_sizes:
        for context_length in args.context_lengths:
            speedup = timing[(t.bfloat16, batch_size, context_length)] / timing[(FP8, batch_size, context_length)]
            print(f"{batch_size:>5} {context_length:>7} {speedup:>11.2f}x")

    print("\nfit: time = overhead + kv_bytes / bandwidth")
    for dtype in names:
        points = [(kv_bytes, ms * 1e-3) for d, _, _, kv_bytes, ms in rows if d == dtype]
        design = t.tensor([[1.0, b] for b, _ in points], dtype=t.float64)
        seconds = t.tensor([[s] for _, s in points], dtype=t.float64)
        overhead, slope = t.linalg.lstsq(design, seconds).solution.flatten().tolist()
        bandwidth = (1 / slope) / 1e9 if slope > 0 else float("nan")
        print(f"  {names[dtype]:>4}: overhead {overhead * 1e6:7.1f} us   bandwidth {bandwidth:6.0f} GB/s")


if __name__ == "__main__":
    main()
