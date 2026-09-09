from torch_llm.kernals.decode_attention import decode_attention_wrapper
import torch as t
import math
from einops import rearrange

def test_decode_attention():
    b = 3
    hq = 8
    hkv = 2
    d = 16
    cache_max_seq_len = 160
    seq_lens = t.tensor([3, 73, 141], dtype=t.int32, device='cuda')

    k_cache = t.randn((b, cache_max_seq_len, hkv, d), dtype=t.float32, device='cuda')
    v_cache = t.randn((b, cache_max_seq_len, hkv, d), dtype=t.float32, device='cuda')
    q = t.rand((b, hq, d), dtype=t.float32, device='cuda')

    actual = decode_attention_wrapper(q, k_cache, v_cache, seq_lens)

    scale = 1 / math.sqrt(d)
    output = t.empty((b, hq, d), dtype=t.float32, device='cuda')

    for batch_idx in range(b):
        q_temp = q[batch_idx].unsqueeze(1)
        k_temp = t.transpose(k_cache[batch_idx][:seq_lens[batch_idx]], dim0=0, dim1=1).repeat_interleave(hq // hkv, 0)
        v_temp = t.transpose(v_cache[batch_idx][:seq_lens[batch_idx]], dim0=0, dim1=1).repeat_interleave(hq // hkv, 0)
        s_tile = q_temp @ k_temp.transpose(-1, -2) * scale
        p_tile = t.softmax(s_tile, dim=-1)
        output[batch_idx] = (p_tile @ v_temp).squeeze(1)

    t.testing.assert_close(actual, output, rtol=1e-2, atol=1e-2)
