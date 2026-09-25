import math

import pytest
import torch as t

from torch_llm.inference.cache_manager import KVCacheManager
from torch_llm.inference.paged_kv_cache import PagedKVCache
from torch_llm.kernals.decode_attention import decode_attention_wrapper


@pytest.mark.skipif(not t.cuda.is_available(), reason="Paged attention requires CUDA")
@t.inference_mode()
@pytest.mark.parametrize("dtype", [t.float32, t.float16, t.bfloat16])
def test_decode_attention(dtype):
    t.manual_seed(4)
    lengths = [3, 73, 141]
    hq, hkv, d = 8, 2, 16
    manager = KVCacheManager(32, 16, 5, 160, "cuda")
    cache = PagedKVCache(32, 16, hkv, d, "cuda", dtype)
    slots, keys, values = [], [], []
    for request_id, length in enumerate(lengths):
        slots.append(manager.allocate_request(request_id, length))
        k = t.randn(length, hkv, d, dtype=dtype, device="cuda")
        v = t.randn_like(k)
        manager.append_request(request_id, t.arange(length, device="cuda"), k, v, cache)
        manager.advance(request_id, length - 1)
        keys.append(k)
        values.append(v)
    slots = t.tensor(slots, device="cuda")
    context = manager.create_container(
        slots, t.arange(4, dtype=t.int32, device="cuda"),
        t.tensor(lengths, device="cuda") - 1,
    )
    q = t.randn(3, hq, d, dtype=dtype, device="cuda")
    actual = decode_attention_wrapper(q, cache, context)
    expected = []
    for row, (k, v) in enumerate(zip(keys, values)):
        k = k.transpose(0, 1).float().repeat_interleave(hq // hkv, 0)
        v = v.transpose(0, 1).float().repeat_interleave(hq // hkv, 0)
        scores = q[row].float().unsqueeze(1) @ k.transpose(-1, -2) / math.sqrt(d)
        expected.append((scores.softmax(-1) @ v).squeeze(1))
    t.testing.assert_close(actual.float(), t.stack(expected), rtol=1e-2, atol=1e-2)
