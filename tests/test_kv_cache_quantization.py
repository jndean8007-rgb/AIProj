"""Tests for the FP8 KV cache (handoff D8, D11, D12).

What is covered:
  A. A PyTorch reference quantizer, defined here in the test file. It is the oracle
     the Triton append kernel is compared against. It is test code, not production
     code: the production cache has no CPU FP8 path.
  B. KVCacheConfig validation (D11, D12).
  C. PagedKVCache storage contract: unquantized vs quantized layout (D11).
  D. Decode refuses an FP8 cache until the decode kernel applies scales (D8 step 4).
  E. The Triton FP8 append kernel against the reference (CUDA, sm89+ only).

Quantization contract (per token, per KV head, over head_dim D):
    amax  = max |x|                       (computed in float32)
    scale = amax / 448  if amax > 0 else 1.0
    q     = cast_fp8(clamp(x / scale, -448, 448))
    x_hat = float(q) * scale
448 is the largest finite float8_e4m3fn value. e4m3fn has no infinity, so an
unclamped overflow becomes NaN, which is why the clamp is part of the contract.

Error bound for x_hat, elementwise:
    |x - x_hat| <= 2**-4 * |x| + scale * 2**-9
e4m3 keeps 3 mantissa bits, so round-to-nearest is off by at most half a step,
2**-4 relative. The scale * 2**-9 term covers values that land in e4m3's
subnormal range (step 2**-9 there, so half a step is 2**-10; 2**-9 gives slack).

Production imports happen inside the tests on purpose, so that sections A and B
still run while `paged_kv_cache.py` fails to import.
"""

import math
from types import SimpleNamespace

import pytest
import torch as t

from torch_llm.inference.kvcache_config import KVCacheConfig


FP8 = t.float8_e4m3fn
FP8_MAX = t.finfo(FP8).max  # 448.0

HAS_FP8_CUDA = t.cuda.is_available() and t.cuda.get_device_capability() >= (8, 9)
requires_fp8_cuda = pytest.mark.skipif(
    not HAS_FP8_CUDA, reason="FP8 Triton casts need a CUDA GPU with compute capability >= 8.9 (Ada/Hopper+)"
)


# ---------------------------------------------------------------------------
# Reference implementation (test oracle)
# ---------------------------------------------------------------------------

def reference_quantize(x):
    """x: [..., D] any float dtype -> (codes [..., D] float8_e4m3fn, scales [...] float32)."""
    x32 = x.float()
    amax = x32.abs().amax(dim=-1)
    scales = t.where(amax > 0, amax / FP8_MAX, t.ones_like(amax))
    codes = (x32 / scales.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return codes, scales


def dequantize(codes, scales):
    return codes.float() * scales.unsqueeze(-1)


def error_bound(x, scales):
    return 2**-4 * x.float().abs() + scales.unsqueeze(-1) * 2**-9


def scattered_locations(num_tokens, num_blocks, block_size, device, seed=0):
    """Distinct (block, offset) pairs spread over the whole pool, in random order."""
    generator = t.Generator().manual_seed(seed)
    flat = t.randperm(num_blocks * block_size, generator=generator)[:num_tokens]
    return (flat // block_size).to(device), (flat % block_size).to(device)


def cache_contents(cache, blocks, offsets):
    """Dequantized K and V at the given locations: two [T, H, D] float32 tensors."""
    k = dequantize(cache.k[blocks, offsets], cache.k_scales[blocks, offsets])
    v = dequantize(cache.v[blocks, offsets], cache.v_scales[blocks, offsets])
    return k, v


# ---------------------------------------------------------------------------
# A. Reference quantizer sanity (proves the oracle is right; CPU)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [t.float32, t.float16, t.bfloat16])
@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 1e3])
def test_reference_round_trip_within_e4m3_bound(dtype, magnitude):
    t.manual_seed(0)
    x = (t.randn(64, 4, 128) * magnitude).to(dtype)
    codes, scales = reference_quantize(x)

    assert codes.dtype == FP8 and codes.shape == x.shape
    assert scales.dtype == t.float32 and scales.shape == x.shape[:-1]
    assert (scales > 0).all()

    error = (x.float() - dequantize(codes, scales)).abs()
    assert (error <= error_bound(x, scales)).all()


def test_reference_largest_element_maps_to_fp8_max():
    x = t.tensor([[0.1, -3.0, 2.0, 0.5]])
    codes, scales = reference_quantize(x)
    assert codes.float()[0, 1].item() == -FP8_MAX
    t.testing.assert_close(dequantize(codes, scales)[0, 1], t.tensor(-3.0))


def test_reference_zero_vector_has_unit_scale_and_zero_codes():
    codes, scales = reference_quantize(t.zeros(3, 2, 16))
    assert (scales == 1.0).all()
    assert (codes.float() == 0).all()


def test_reference_outlier_channel_still_within_bound():
    # Per-token scales are set by the largest channel, so small channels lose
    # precision when one channel is an outlier (common for K after RoPE).
    t.manual_seed(1)
    x = t.randn(32, 2, 64)
    x[..., 7] *= 1000
    codes, scales = reference_quantize(x)
    error = (x - dequantize(codes, scales)).abs()
    assert (error <= error_bound(x, scales)).all()


def test_reference_never_produces_nan_or_inf():
    x = t.cat([
        t.randn(16, 2, 32) * 1e30,
        t.randn(16, 2, 32) * 1e-30,
        t.full((16, 2, 32), 65504.0),  # float16 max
    ])
    codes, scales = reference_quantize(x)
    assert t.isfinite(codes.float()).all()
    assert t.isfinite(scales).all()


# ---------------------------------------------------------------------------
# B. KVCacheConfig (D11, D12; CPU)
# ---------------------------------------------------------------------------

def test_config_defaults_to_unquantized_bf16():
    config = KVCacheConfig()
    assert config.kv_cache_dtype == t.bfloat16
    assert config.kv_scale_dtype == t.float32
    assert config.scale_granularity == "token_head"


@pytest.mark.parametrize("dtype", [t.float16, t.bfloat16, t.float32, FP8])
def test_config_accepts_supported_cache_dtypes(dtype):
    assert KVCacheConfig(kv_cache_dtype=dtype).kv_cache_dtype == dtype


@pytest.mark.parametrize("field,value", [
    ("num_blocks", 0),
    ("num_blocks", -4),
    ("block_size", 0),
    ("block_size", 3),
    ("max_cache_slots", 0),
    ("cache_max_seq_len", -1),
    ("num_blocks", True),          # bool is an int subclass; must still be rejected
    ("block_size", 16.0),
    ("kv_cache_dtype", t.int8),
    ("kv_cache_dtype", t.float8_e5m2),
    ("kv_scale_dtype", t.float16),
    ("scale_granularity", "per_block"),
])
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        KVCacheConfig(**{field: value})


# ---------------------------------------------------------------------------
# C. PagedKVCache storage contract (D11; CPU)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [t.float16, t.bfloat16, t.float32])
def test_unquantized_cache_has_no_scales(dtype):
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    cache = PagedKVCache(6, 4, 2, 16, "cpu", dtype)  # positional dtype, as existing tests use
    assert cache.quantized is False
    assert cache.k.shape == cache.v.shape == (6, 4, 2, 16)
    assert cache.k.dtype == cache.v.dtype == dtype
    assert cache.k_scales is None and cache.v_scales is None


def test_fp8_cache_allocates_fp32_scales_per_token_and_head():
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    cache = PagedKVCache(6, 4, 2, 16, "cpu", FP8)
    assert cache.quantized is True
    assert cache.k.shape == cache.v.shape == (6, 4, 2, 16)
    assert cache.k.dtype == cache.v.dtype == FP8
    assert cache.k_scales.shape == cache.v_scales.shape == (6, 4, 2)
    assert cache.k_scales.dtype == cache.v_scales.dtype == t.float32


@pytest.mark.parametrize("dtype,quantized", [(t.bfloat16, False), (FP8, True)])
def test_initialize_cache_follows_config_dtype(dtype, quantized):
    from torch_llm.inference.cache_manager import initialize_cache

    config = KVCacheConfig(num_blocks=8, block_size=4, max_cache_slots=2, cache_max_seq_len=16, kv_cache_dtype=dtype)
    model_config = SimpleNamespace(num_layers=3, num_kv_heads=2, head_dim=16)
    _, caches = initialize_cache(config, model_config, "cpu")

    assert len(caches) == 3
    assert len({id(cache.k) for cache in caches}) == 3  # independent storage per layer
    for cache in caches:
        assert cache.quantized is quantized
        assert cache.k.dtype == dtype


@pytest.mark.parametrize("dtype", [t.float16, t.bfloat16, t.float32])
def test_unquantized_append_writes_only_targeted_slots(dtype):
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    cache = PagedKVCache(6, 4, 2, 16, "cpu", dtype)
    cache.k.fill_(-7.0)
    cache.v.fill_(-7.0)
    blocks, offsets = scattered_locations(9, 6, 4, "cpu")
    ks = t.randn(9, 2, 16)  # float32 input; the cache casts to its own dtype
    vs = t.randn(9, 2, 16)

    cache.append_kv(blocks, offsets, ks, vs)

    t.testing.assert_close(cache.k[blocks, offsets], ks.to(dtype))
    t.testing.assert_close(cache.v[blocks, offsets], vs.to(dtype))
    written = t.zeros(6, 4, dtype=t.bool)
    written[blocks, offsets] = True
    assert (cache.k[~written] == -7.0).all()
    assert (cache.v[~written] == -7.0).all()


# ---------------------------------------------------------------------------
# D. Decode must refuse an FP8 cache until it applies scales (CPU)
# ---------------------------------------------------------------------------

def test_decode_rejects_quantized_cache_until_implemented():
    # Reading FP8 codes without their scales gives plausible-looking garbage,
    # so the wrapper must raise before touching the batch context or launching.
    from torch_llm.inference.paged_kv_cache import PagedKVCache
    from torch_llm.kernals.decode_attention import decode_attention_wrapper

    cache = PagedKVCache(2, 4, 2, 16, "cpu", FP8)
    q = t.randn(1, 4, 16)
    with pytest.raises(NotImplementedError):
        decode_attention_wrapper(q, cache, None)


# ---------------------------------------------------------------------------
# E. Triton FP8 append kernel vs reference (CUDA sm89+)
# ---------------------------------------------------------------------------

def assert_matches_reference(cache, blocks, offsets, ks, vs):
    """Scales match the reference; dequantized values are within one FP8 step.

    Codes may differ from the reference in rare round-to-nearest ties, because
    Triton's float32 division isn't guaranteed to be bit-identical to PyTorch's.
    """
    for x, codes, scales in (
        (ks, cache.k[blocks, offsets], cache.k_scales[blocks, offsets]),
        (vs, cache.v[blocks, offsets], cache.v_scales[blocks, offsets]),
    ):
        ref_codes, ref_scales = reference_quantize(x)
        t.testing.assert_close(scales, ref_scales, rtol=1e-6, atol=0)

        actual = dequantize(codes, scales)
        expected = dequantize(ref_codes, ref_scales)
        assert t.isfinite(actual).all()
        one_step = 2**-3 * expected.abs() + ref_scales.unsqueeze(-1) * 2**-8
        assert ((actual - expected).abs() <= one_step).all()
        mismatch_fraction = (codes.float() != ref_codes.float()).float().mean().item()
        assert mismatch_fraction <= 1e-2
        assert ((x.float() - actual).abs() <= 2 * error_bound(x, ref_scales)).all()


@requires_fp8_cuda
@t.inference_mode()
@pytest.mark.parametrize("input_dtype", [t.float32, t.float16, t.bfloat16])
@pytest.mark.parametrize("head_dim", [16, 80, 128])  # 80 is not a power of two: exercises the D mask
def test_fp8_append_matches_reference(input_dtype, head_dim):
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    t.manual_seed(2)
    num_blocks, block_size, num_kv_heads, num_tokens = 16, 16, 4, 100
    cache = PagedKVCache(num_blocks, block_size, num_kv_heads, head_dim, "cuda", FP8)
    blocks, offsets = scattered_locations(num_tokens, num_blocks, block_size, "cuda")
    ks = (t.randn(num_tokens, num_kv_heads, head_dim, device="cuda") * 3).to(input_dtype)
    vs = t.randn(num_tokens, num_kv_heads, head_dim, device="cuda").to(input_dtype)

    cache.append_kv(blocks, offsets, ks, vs)

    assert_matches_reference(cache, blocks, offsets, ks, vs)


@requires_fp8_cuda
@t.inference_mode()
def test_fp8_append_writes_only_targeted_slots():
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    cache = PagedKVCache(8, 4, 2, 32, "cuda", FP8)
    cache.k.copy_(t.full(cache.k.shape, 5.0, device="cuda").to(FP8))
    cache.v.copy_(t.full(cache.v.shape, 5.0, device="cuda").to(FP8))
    cache.k_scales.fill_(-1.0)  # real scales are always positive
    cache.v_scales.fill_(-1.0)
    blocks, offsets = scattered_locations(11, 8, 4, "cuda")

    cache.append_kv(blocks, offsets, t.randn(11, 2, 32, device="cuda"), t.randn(11, 2, 32, device="cuda"))

    written = t.zeros(8, 4, dtype=t.bool, device="cuda")
    written[blocks, offsets] = True
    for storage in (cache.k, cache.v):
        assert (storage[~written].float() == 5.0).all()
    for scales in (cache.k_scales, cache.v_scales):
        assert (scales[~written] == -1.0).all()
        assert (scales[written] > 0).all()


@requires_fp8_cuda
@t.inference_mode()
def test_fp8_append_accepts_non_contiguous_input():
    # The unquantized path accepts any strided [T, H, D] via indexing; the FP8
    # path must too, rather than reading the wrong elements.
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    t.manual_seed(3)
    cache = PagedKVCache(8, 4, 2, 32, "cuda", FP8)
    blocks, offsets = scattered_locations(10, 8, 4, "cuda")
    ks = t.randn(2, 10, 32, device="cuda").transpose(0, 1)  # [10, 2, 32], non-contiguous
    vs = t.randn(2, 10, 32, device="cuda").transpose(0, 1)
    assert not ks.is_contiguous()

    cache.append_kv(blocks, offsets, ks, vs)

    assert_matches_reference(cache, blocks, offsets, ks, vs)


@requires_fp8_cuda
@t.inference_mode()
@pytest.mark.parametrize("input_dtype,peak", [(t.float32, 1e30), (t.float32, 1e-30), (t.float16, 65504.0), (t.bfloat16, 1e30)])
def test_fp8_append_extreme_magnitudes_stay_finite(input_dtype, peak):
    # Each (token, head) vector is rescaled so its largest element is exactly `peak`
    # (65504 is the float16 maximum), so the input itself is always finite.
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    t.manual_seed(4)
    cache = PagedKVCache(8, 8, 2, 64, "cuda", FP8)
    blocks, offsets = scattered_locations(40, 8, 8, "cuda")
    x = t.randn(40, 2, 64, device="cuda", dtype=t.float64)
    x = (x / x.abs().amax(-1, keepdim=True) * peak).to(input_dtype)
    assert t.isfinite(x).all()

    cache.append_kv(blocks, offsets, x, x)

    k, v = cache_contents(cache, blocks, offsets)
    assert t.isfinite(k).all() and t.isfinite(v).all()
    assert (cache.k_scales[blocks, offsets] > 0).all()


@requires_fp8_cuda
@t.inference_mode()
def test_fp8_append_zero_vectors():
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    cache = PagedKVCache(4, 4, 2, 16, "cuda", FP8)
    blocks, offsets = scattered_locations(5, 4, 4, "cuda")
    zeros = t.zeros(5, 2, 16, device="cuda")

    cache.append_kv(blocks, offsets, zeros, zeros)

    assert (cache.k_scales[blocks, offsets] == 1.0).all()
    assert (cache.k[blocks, offsets].float() == 0).all()


@requires_fp8_cuda
@t.inference_mode()
def test_fp8_append_through_cache_manager_round_trips():
    # Same entry point the runtime uses: manager resolves locations, cache stores.
    from torch_llm.inference.cache_manager import KVCacheManager
    from torch_llm.inference.paged_kv_cache import PagedKVCache

    t.manual_seed(5)
    manager = KVCacheManager(16, 4, 2, 32, "cuda")
    cache = PagedKVCache(16, 4, 2, 32, "cuda", FP8)
    length = 13
    slot = manager.allocate_request(0, length)
    ks = t.randn(length, 2, 32, device="cuda", dtype=t.bfloat16)
    vs = t.randn_like(ks)

    manager.append_request(0, t.arange(length, device="cuda"), ks, vs, cache)

    positions = t.arange(length, device="cuda")
    blocks = manager.block_table[slot, positions // 4]
    offsets = positions % 4
    k, v = cache_contents(cache, blocks, offsets)
    _, k_scales = reference_quantize(ks)
    _, v_scales = reference_quantize(vs)
    assert ((ks.float() - k).abs() <= 2 * error_bound(ks, k_scales)).all()
    assert ((vs.float() - v).abs() <= 2 * error_bound(vs, v_scales)).all()
