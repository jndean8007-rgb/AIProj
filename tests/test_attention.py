import math

import pytest
import torch as t

from torch_llm.inference.cache_manager import KVCacheManager
from torch_llm.inference.paged_kv_cache import PagedKVCache
from torch_llm.model.attention import Attention


pytestmark = pytest.mark.skipif(not t.cuda.is_available(), reason="Attention kernels require CUDA")


@pytest.fixture
def attention():
    t.manual_seed(1)
    return Attention(128, 8, 2, 16, 256).cuda().eval()


@t.inference_mode()
def test_attention_train_mode(attention):
    x = t.randn(14, 128, device="cuda")
    output = attention(
        x, t.arange(7, device="cuda").repeat(2),
        t.tensor([0, 7, 14], dtype=t.int32, device="cuda"), 7, mode="train",
    )
    assert output.shape == x.shape
    assert t.isfinite(output).all()


@t.inference_mode()
def test_attention_prefill_updates_cache(attention):
    manager = KVCacheManager(8, 4, 4, 16, "cuda")
    cache = PagedKVCache(8, 4, 2, 16, "cuda", t.float32)
    slots = t.tensor([manager.allocate_request(10, 3), manager.allocate_request(20, 5)], device="cuda")
    cu = t.tensor([0, 3, 8], dtype=t.int32, device="cuda")
    positions = t.tensor([0, 1, 2, 0, 1, 2, 3, 4], device="cuda")
    x = t.randn(8, 128, device="cuda")
    context = manager.create_container(slots, cu, positions)
    expected = attention(x, positions, cu, 5, mode="train")
    actual = attention(x, positions, cu, 5, mode="prefill", paged_kv_cache=cache, cache_batch_context=context)
    manager.advance_batch([10, 20], [3, 5])
    t.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert manager.seq_lens[slots].tolist() == [3, 5]
    expected_k = attention.rope(attention.k_proj(x).reshape(8, 2, 16), positions)
    expected_v = attention.v_proj(x).reshape(8, 2, 16)
    t.testing.assert_close(cache.k[context.physical_blocks, context.block_offsets], expected_k)
    t.testing.assert_close(cache.v[context.physical_blocks, context.block_offsets], expected_v)


@t.inference_mode()
def test_attention_decode_updates_cache_and_matches_reference(attention):
    manager = KVCacheManager(8, 4, 4, 16, "cuda")
    cache = PagedKVCache(8, 4, 2, 16, "cuda", t.float32)
    lengths = [3, 5]
    slots = []
    for request_id, length in enumerate(lengths):
        slots.append(manager.allocate_request(request_id, length + 1))
        k = t.randn(length, 2, 16, device="cuda")
        v = t.randn_like(k)
        manager.append_request(request_id, t.arange(length, device="cuda"), k, v, cache)
        manager.advance(request_id, length)
    slots = t.tensor(slots, device="cuda")
    positions = t.tensor(lengths, device="cuda")
    cu = t.arange(3, dtype=t.int32, device="cuda")
    context = manager.create_container(slots, cu, positions)
    x = t.randn(2, 128, device="cuda")
    actual = attention(x, positions, cu, 1, mode="decode", paged_kv_cache=cache, cache_batch_context=context)
    manager.advance_batch([0, 1], [1, 1])
    q = attention.rope(attention.q_proj(x).reshape(2, 8, 16), positions)
    expected_k = attention.rope(attention.k_proj(x).reshape(2, 2, 16), positions)
    t.testing.assert_close(cache.k[context.physical_blocks, context.block_offsets], expected_k)
    t.testing.assert_close(cache.v[context.physical_blocks, context.block_offsets], attention.v_proj(x).reshape(2, 2, 16))
    assert manager.seq_lens[slots].tolist() == [4, 6]
    rows = []
    for row, (slot, length) in enumerate(zip(slots, lengths)):
        p = t.arange(length + 1, device="cuda")
        blocks = manager.block_table[slot, p // 4]
        k = cache.k[blocks, p % 4].transpose(0, 1).repeat_interleave(4, 0)
        v = cache.v[blocks, p % 4].transpose(0, 1).repeat_interleave(4, 0)
        scores = q[row].unsqueeze(1) @ k.transpose(-1, -2) / math.sqrt(16)
        rows.append((scores.softmax(-1) @ v).reshape(-1))
    expected = attention.o_proj(t.stack(rows))
    t.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
