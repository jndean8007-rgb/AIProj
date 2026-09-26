import torch as t
import math
import triton
import triton.language as tl


'''
token × KV head

load Dh values
→ max(abs(...))
→ calculate scale
→ quantize Dh values
→ store quantized vector
→ store one scale

k[physical_block, offset, head, :]
k_scales[physical_block, offset, head]

v[physical_block, offset, head, :]
v_scales[physical_block, offset, head]
'''

'''
contract:
loads in head dimension values for a particular block, block offset, head.
gets max value. calculates scale: scale = max abs d / t.finfo(cache_dtype).ax
p = quantize(x / s)

x^ = p * s

write both x^ and scale
'''

def quantize_append_wrapper(
        paged_kv_cache,
        ks,
        vs,
        physical_blocks,
        block_offsets,
):
    assert ks.shape == vs.shape
    assert paged_kv_cache.k.shape == paged_kv_cache.v.shape
    assert ks.shape[-2:] == paged_kv_cache.k.shape[-2:]

    num_blocks, block_size, num_kv_heads, head_dim = paged_kv_cache.k.shape
    num_tokens = ks.shape[0]

    cache_dtype = paged_kv_cache.k.dtype
    max_rep_cache_dtype = t.finfo(cache_dtype).max

    block_d = triton.next_power_of_2(head_dim)
    grid = (num_tokens, num_kv_heads)

    quantize_append[grid](
        ks,
        vs,
        paged_kv_cache.k,
        paged_kv_cache.v,
        paged_kv_cache.k_scales,
        paged_kv_cache.v_scales,

        physical_blocks,
        block_offsets,
        head_dim,
        num_kv_heads,

        cache_dtype,
        max_rep_cache_dtype,
        block_d,
        block_size,
    )

@triton.jit
def quantize_append(
        k_ptr,
        v_ptr,
        k_cache_ptr,
        v_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,

        physical_block_ptr,
        block_offsets_ptr,
        head_dim,
        num_kv_heads,
        cache_dtype,

        MAX_REP_CACHE_DTYPE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
): # each kernel launch owns ONE token, ONE head.

    global_token_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)

    physical_block = tl.load(physical_block_ptr + global_token_id)
    block_offset = tl.load(block_offsets_ptr + global_token_id)

    head_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < head_dim


    kv_pos = (global_token_id * num_kv_heads + kv_head_id) * head_dim + head_offsets
    kv_cache_pos = ((physical_block * BLOCK_SIZE + block_offset) * num_kv_heads + kv_head_id) * head_dim + head_offsets
    kv_scale_pos = ((physical_block * BLOCK_SIZE + block_offset) * num_kv_heads + kv_head_id)

    k_tile = tl.load(
        k_ptr + kv_pos,
        mask=head_mask,
        other=0.0,
    )

    k_max = tl.max(tl.abs(k_tile), axis=-1)

    v_tile = tl.load(
        v_ptr + kv_pos,
        mask=head_mask,
        other=0.0,
    )

    v_max = tl.max(tl.abs(v_tile), axis=-1)

    k_scale = tl.where(k_max > 0, k_max / MAX_REP_CACHE_DTYPE, 1.0)
    v_scale = tl.where(v_max > 0, v_max / MAX_REP_CACHE_DTYPE, 1.0)

    tl.store(k_scale_ptr + kv_scale_pos, k_scale)
    tl.store(v_scale_ptr + kv_scale_pos, v_scale)

    quantized_ks = tl.cast((k_tile / k_scale), dtype=cache_dtype)
    quantized_vs = tl.cast((v_tile / v_scale), dtype=cache_dtype)

    tl.store(k_cache_ptr + kv_cache_pos, quantized_ks)
    tl.store(v_cache_ptr + kv_cache_pos, quantized_vs)



