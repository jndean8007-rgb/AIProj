import torch as t
import triton
import triton.language as tl
import math
from jaxtyping import Shaped

#need wrapper
#time to turn attention into new caches
def decode_attention_wrapper(
        q,
        paged_kv_cache,
        cache_batch_context
):
    b, hq, d = q.shape
    scale = q / math.sqrt(b)
    #each cache slot + block table + context length helps resolve which tokens are relevant

    #b corresponds to cache slots
    #tps derived from num splits and context lens
    #exact location derived from aggregate, including block table

    k_cache = paged_kv_cache.k
    v_cache = paged_kv_cache.v
    hkv = v_cache.shape[-2]

    assert hq % hkv == 0

    cache_slots = cache_batch_context.cache_slots
    context_lengths = cache_batch_context.context_lengths #already in reduced form, unlike block table
    block_table = cache_batch_context.block_table[cache_slots]

    BLOCKS_PER_SPLIT = 4 #tuning variable

    BLOCK_SIZE = paged_kv_cache.block_size
    blocks_per_sequence = (context_lengths + BLOCK_SIZE - 1) // BLOCK_SIZE
    splits_per_sequence = (blocks_per_sequence + BLOCKS_PER_SPLIT - 1) // BLOCKS_PER_SPLIT

    cum_splits = t.cat([
        t.zeros(1, dtype=t.long, device=q.device),
        t.cumsum(splits_per_sequence, dim=0)]
    )


    num_splits = 8

    partial_max = t.empty((b, hq, num_splits), dtype=t.float32, device=q.device)
    partial_sum = t.empty((b, hq, num_splits), dtype=t.float32, device=q.device)
    partial_accum = t.empty((b, hq, num_splits, d), dtype=t.float32, device=q.device)

    BLOCK_N = 64
    HEAD_DIM = triton.next_power_of_2(d)
    grid = (b, hq, num_splits)

    decode_attention_split_kernal[grid](
        q,
        k_cache,
        v_cache,

        partial_max,
        partial_sum,
        partial_accum,

        cache_slots,
        context_lengths,
        block_table,
        cum_splits,

        d,
        hq,
        hkv,
        scale,

        num_splits,
        BLOCK_N,
        HEAD_DIM,
    )

    )



def decode_attention_wrapper(q: Shaped[t.Tensor, 'b hq d'],
                             k_cache: Shaped[t.Tensor, 'b s hkv d'],
                             v_cache: Shaped[t.Tensor, 'b s hkv d'],
                             seq_lens: Shaped[t.Tensor, 'b']):

    assert k_cache.shape == v_cache.shape
    assert q.shape[0] == v_cache.shape[0]
    assert q.shape[2] == v_cache.shape[3]
    assert seq_lens.shape[0] == q.shape[0]
    assert seq_lens.dtype == t.int32
    assert seq_lens.device == k_cache.device == q.device == v_cache.device
    assert all(k_cache.shape[1] > seq_lens[b] >= 1 for b in range(q.shape[0])) # cache capacity and nonzero active lengths
    assert k_cache.is_contiguous() and v_cache.is_contiguous() and q.is_contiguous()

    b, hq, d = q.shape
    scale = 1 / math.sqrt(d)

    cache_max_seq_len = v_cache.shape[1]
    hkv = v_cache.shape[-2]

    assert hq % hkv == 0
    num_splits = 8 #later can choose number based on varous other details

    partial_max = t.empty((b, hq, num_splits), dtype=t.float32, device=q.device)
    partial_sum = t.empty((b, hq, num_splits), dtype=t.float32, device=q.device)
    partial_accum = t.empty((b, hq, num_splits, d), dtype=t.float32, device=q.device)

    BLOCK_N = 64
    HEAD_DIM = triton.next_power_of_2(d)
    grid = (b, hq, num_splits)

    decode_attention_split_kernal[grid](
        q,
        k_cache,
        v_cache,
        seq_lens,

        partial_max,
        partial_sum,
        partial_accum,

        d,
        hq,
        hkv,
        cache_max_seq_len,
        scale,

        num_splits,
        BLOCK_N,
        HEAD_DIM,
    ) # retrieve partial accums, partial maxes, and partial sums for each batch head and token split in chunk of 64

    output = t.empty((b, hq, d), dtype=q.dtype, device=q.device)

    #run through reduction kernal to combine partials and write to output
    grid = (b, hq)
    decode_attention_reduction_kernal[grid](
        partial_max,
        partial_sum,
        partial_accum,
        output,

        d,
        hq,

        num_splits,
        HEAD_DIM,
    )

    return output

    #distinguish between cache capacity and valid cache length for each request


@triton.jit
def decode_attention_split_kernal(
        q_ptr,
        kcache_ptr,
        vcache_ptr,
        seq_lens_ptr,

        pmax_ptr,
        psum_ptr,
        paccum_ptr,

        d,
        hq,
        hkv,
        cache_max_seq_len,
        scale,

        num_splits: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_head_id = tl.program_id(1)
    split_id = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + batch_id)
    tps = (seq_len + num_splits - 1) // num_splits
    qh_per_kvh = hq // hkv

    split_start = split_id * tps
    split_end = min(split_start + tps, seq_len)

    kv_head_id = query_head_id // qh_per_kvh # simply which head is it, not memory #(B, cache_max_seq_len, Hkv, D) is cache layout

    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < d

    #q is batch id hq, + query head id
    q_pos = (batch_id * hq + query_head_id) * d + head_offsets
    q_tile = tl.load(q_ptr + q_pos, mask=head_mask, other=0.0) #single batch, single head, 1 token, and q_pos extends further than headdim for memory padding purposes

    partial_max = tl.cast(-float("inf"), tl.float32)
    partial_sum = tl.cast(0.0, tl.float32)
    partial_accum = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for kv_start in tl.range(split_start, split_end, BLOCK_N):
        kv_offsets = kv_start + tl.arange(0, BLOCK_N) #
        kv_mask = kv_offsets < split_end
        kv_row_base = ((batch_id * cache_max_seq_len + kv_offsets) * hkv + kv_head_id) * d
        kv_pos = kv_row_base[:, None] + head_offsets[None, :]
        load_mask = kv_mask[:, None] & head_mask[None, :]

        k_tile = tl.load(kcache_ptr + kv_pos, mask=load_mask, other=0.0)
        v_tile = tl.load(vcache_ptr + kv_pos, mask=load_mask, other=0.0)

        s_tile = scale * tl.squeeze(tl.dot(k_tile, q_tile[:, None]), dim = 1)
        s_tile = tl.where(kv_mask, s_tile, -float("inf"))
        tile_max = tl.max(s_tile, axis=-1)
        old_max = partial_max
        new_max = tl.maximum(old_max, tile_max)
        alpha = tl.exp(old_max - new_max)
        p = tl.exp(s_tile - new_max)
        partial_sum = alpha * partial_sum + tl.sum(p, axis=-1)
        partial_accum = tl.squeeze(alpha * partial_accum + tl.dot(p[None, :], v_tile.to(p.dtype)), dim = 0)

        partial_max = new_max

    pmax_pos = (batch_id * hq + query_head_id) * num_splits + split_id
    psum_pos = pmax_pos
    tl.store(pmax_ptr + pmax_pos, partial_max)
    tl.store(psum_ptr + psum_pos, partial_sum)
    paccum_pos = ((batch_id * hq + query_head_id) * num_splits + split_id) * d + head_offsets
    tl.store(paccum_ptr + paccum_pos, partial_accum, mask=head_mask)

@triton.jit
def decode_attention_reduction_kernal(
        pmax_ptr,
        psum_ptr,
        paccum_ptr,
        output_ptr,

        head_dim,
        hq,

        num_splits: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_head_id = tl.program_id(1)

    split_offset = tl.arange(0, num_splits)
    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < head_dim

    p_max_pos = (batch_id * hq + query_head_id) * num_splits + split_offset
    p_maxes = tl.load(pmax_ptr + p_max_pos)
    global_max = tl.max(p_maxes, axis=0)
    corrections = tl.exp(p_maxes - global_max)

    p_sum_pos = p_max_pos
    p_sums = tl.load(psum_ptr + p_sum_pos)
    denominator = tl.sum(corrections * p_sums, axis=-1)

    p_accum_pos = ((batch_id * hq + query_head_id) * num_splits + split_offset[:, None]) * head_dim + head_offsets[None, :]
    p_accums = tl.load(paccum_ptr + p_accum_pos, mask=head_mask[None, :])
    numerator = tl.sum(
        corrections[:, None] * p_accums,
        axis=-2
    )

    output = numerator / denominator
    output_pos = (batch_id * hq + query_head_id) * head_dim + head_offsets
    tl.store(output_ptr + output_pos, output, mask=head_mask)
