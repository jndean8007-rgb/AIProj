import math

import pytest
import torch as t
from einops import rearrange

from torch_llm.inference.kv_cache import KVCache
from torch_llm.kernals.decode_attention import decode_attention_wrapper
from torch_llm.model.attention import Attention


@pytest.fixture
def attention():
    module = Attention(
        d_model=128,
        num_q_heads = 8,
        num_kv_heads = 2,
        head_dim = 16,
        model_max_seq_len = 256,
        theta = 10000
    ).cuda()
    return module


def make_dense_packed_metadata(batch_size, seq_len, device="cuda"):
    """
    Dense B x S batch represented in packed form.
    """
    T = batch_size * seq_len

    cu_seqlens = t.arange(
        0,
        T + 1,
        seq_len,
        dtype=t.int32,
        device=device,
    )

    token_positions = (
        t.arange(seq_len, device=device)
        .repeat(batch_size)
    )

    return token_positions, cu_seqlens, seq_len


def test_attention_train_mode(attention):
    """
    Train mode:
      - returns correct packed shape
      - does not require/use a KV cache
    """
    b = 2
    s = 7
    d_model = attention.d_model  # adjust if your class names this differently

    x = t.randn(
        (b * s, d_model),
        device="cuda",
        dtype=t.float32,
    )

    token_positions, cu_seqlens, batch_max_seq_len = (
        make_dense_packed_metadata(b, s)
    )

    output = attention(
        x,
        token_positions,
        cu_seqlens,
        batch_max_seq_len,
        kv_cache=None,
        mode="train",
    )

    assert output.shape == (b * s, d_model)
    assert t.isfinite(output).all()


def test_attention_prefill_updates_cache(attention):
    """
    Prefill mode:
      1. output should equal train-mode attention on the same input
      2. cache lengths should advance by prompt lengths
      3. cached K/V should equal the K/V produced by the attention module
         after RoPE has been applied to K
    """
    lengths = [3, 5]
    b = len(lengths)
    T = sum(lengths)
    d_model = attention.d_model  # adjust name if necessary

    cu_seqlens = t.tensor(
        [0, 3, 8],
        dtype=t.int32,
        device="cuda",
    )

    token_positions = t.tensor(
        [0, 1, 2, 0, 1, 2, 3, 4],
        dtype=t.long,
        device="cuda",
    )

    batch_max_seq_len = max(lengths)

    x = t.randn(
        (T, d_model),
        device="cuda",
        dtype=t.float32,
    )

    cache = KVCache(
        batch_size=b,
        cache_max_seq_len=16,
        num_kv_heads=attention.num_kv_heads,
        head_dim=attention.head_dim,
        dtype=t.float32,
        device="cuda",
    )

    # -------------------
    # Expected K/V
    # -------------------
    with t.no_grad():
        expected_k = attention.k_proj(x)

        expected_k = rearrange(
            expected_k,
            "T (h d) -> T h d",
            h=attention.num_kv_heads,
            d=attention.head_dim,
        )

        expected_k = attention.rope(
            expected_k,
            token_positions,
        )

        expected_v = attention.v_proj(x)

        expected_v = rearrange(
            expected_v,
            "T (h d) -> T h d",
            h=attention.num_kv_heads,
            d=attention.head_dim,
        )

    # Train mode gives the same attention computation,
    # but without mutating a cache.
    expected_output = attention(
        x,
        token_positions,
        cu_seqlens,
        batch_max_seq_len,
        kv_cache=None,
        mode="train",
    )

    actual_output = attention(
        x,
        token_positions,
        cu_seqlens,
        batch_max_seq_len,
        kv_cache=cache,
        mode="prefill",
    )

    t.testing.assert_close(
        actual_output,
        expected_output,
        rtol=1e-2,
        atol=1e-2,
    )

    expected_lengths = t.tensor(
        lengths,
        dtype=t.int32,
        device="cuda",
    )

    t.testing.assert_close(
        cache.seq_lens,
        expected_lengths,
    )

    # Check request 0
    t.testing.assert_close(
        cache.k_cache[0, :3],
        expected_k[0:3],
        rtol=1e-5,
        atol=1e-5,
    )

    t.testing.assert_close(
        cache.v_cache[0, :3],
        expected_v[0:3],
        rtol=1e-5,
        atol=1e-5,
    )

    # Check request 1
    t.testing.assert_close(
        cache.k_cache[1, :5],
        expected_k[3:8],
        rtol=1e-5,
        atol=1e-5,
    )

    t.testing.assert_close(
        cache.v_cache[1, :5],
        expected_v[3:8],
        rtol=1e-5,
        atol=1e-5,
    )


def test_attention_decode_updates_cache_and_matches_decode_kernel(attention):
    """
    Decode mode:
      1. append one new K/V per request
      2. seq_lens increase by one
      3. high-level Attention output equals direct decode kernel output
         using the updated cache
    """
    b = 2
    d_model = attention.d_model  # adjust if needed
    hkv = attention.num_kv_heads
    d = attention.head_dim

    cache = KVCache(
        batch_size=b,
        cache_max_seq_len=16,
        num_kv_heads=hkv,
        head_dim=d,
        dtype=t.float32,
        device="cuda",
    )

    # Pretend prefill has already cached:
    # request 0: 3 tokens
    # request 1: 5 tokens
    cache.seq_lens[:] = t.tensor(
        [3, 5],
        dtype=t.int32,
        device="cuda",
    )

    # Existing cache contents can simply be random for this routing test.
    cache.k_cache[0, :3] = t.randn_like(cache.k_cache[0, :3])
    cache.v_cache[0, :3] = t.randn_like(cache.v_cache[0, :3])

    cache.k_cache[1, :5] = t.randn_like(cache.k_cache[1, :5])
    cache.v_cache[1, :5] = t.randn_like(cache.v_cache[1, :5])

    old_lengths = cache.seq_lens.clone()

    # One packed decode token per request => T == B
    x = t.randn(
        (b, d_model),
        dtype=t.float32,
        device="cuda",
    )

    # New tokens are at the current cache positions.
    token_positions = old_lengths.to(dtype=t.long)

    # One token per request:
    # request 0 -> packed [0:1]
    # request 1 -> packed [1:2]
    cu_seqlens = t.tensor(
        [0, 1, 2],
        dtype=t.int32,
        device="cuda",
    )

    batch_max_seq_len = 1

    # -------------------
    # Resolve Q/K/V manually
    # -------------------
    with t.no_grad():
        q = attention.q_proj(x)
        k = attention.k_proj(x)
        v = attention.v_proj(x)

        q = rearrange(
            q,
            "T (h d) -> T h d",
            h=attention.num_q_heads,
            d=d,
        )

        k = rearrange(
            k,
            "T (h d) -> T h d",
            h=hkv,
            d=d,
        )

        v = rearrange(
            v,
            "T (h d) -> T h d",
            h=hkv,
            d=d,
        )

        q_rope = attention.rope(q, token_positions)
        k_rope = attention.rope(k, token_positions)

    actual = attention(
        x,
        token_positions,
        cu_seqlens,
        batch_max_seq_len,
        kv_cache=cache,
        mode="decode",
    )

    # Cache should now contain the newly generated K/V.
    expected_lengths = old_lengths + 1

    t.testing.assert_close(
        cache.seq_lens,
        expected_lengths,
    )

    batch_indices = t.arange(b, device="cuda")

    # Newly appended token lives at the OLD length index.
    t.testing.assert_close(
        cache.k_cache[batch_indices, old_lengths],
        k_rope,
        rtol=1e-5,
        atol=1e-5,
    )

    t.testing.assert_close(
        cache.v_cache[batch_indices, old_lengths],
        v,
        rtol=1e-5,
        atol=1e-5,
    )

    # Direct low-level decode result after the cache update.
    expected_attention = decode_attention_wrapper(
        q_rope,
        cache.k_cache,
        cache.v_cache,
        cache.seq_lens,
    )

    expected_attention = rearrange(
        expected_attention,
        "T h d -> T (h d)",
    )

    expected = attention.o_proj(expected_attention)

    t.testing.assert_close(
        actual,
        expected,
        rtol=1e-2,
        atol=1e-2,
    )
