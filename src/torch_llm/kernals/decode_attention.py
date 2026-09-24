import torch as t
import triton
import triton.language as tl
import math
from jaxtyping import Shaped


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

    total_splits = cum_splits[-1]

    block_table_width = block_table.shape[1]

    partial_max = t.empty((b, hq, total_splits), dtype=t.float32, device=q.device)
    partial_sum = t.empty((b, hq, total_splits), dtype=t.float32, device=q.device)
    partial_accum = t.empty((b, hq, total_splits, d), dtype=t.float32, device=q.device)

    BLOCK_N = 64
    HEAD_DIM = triton.next_power_of_2(d)
    grid = (b, hq, total_splits)

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

        total_splits,
        BLOCK_SIZE,
        BLOCKS_PER_SPLIT,
        block_table_width,
        BLOCK_N,
        HEAD_DIM,
    )

    output = t.empty((b, hq, d), dtype=q.dtype, device=q.device)

    grid = (b, hq)
    decode_attention_reduction_kernal[grid](
        partial_max,
        partial_sum,
        partial_accum,
        output,

        d,
        hq,

        total_splits,
        HEAD_DIM,
    )

    return output

@triton.jit
def decode_attention_split_kernal(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,

        partial_max_ptr,
        partial_sum_ptr,
        partial_accum_ptr,

        cache_slots_ptr,
        context_lengths_ptr,
        block_table_ptr,
        cum_splits_ptr,

        d,
        hq,
        hkv,
        scale,

        TOTAL_SPLITS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_head_id = tl.program_id(1)
    split_id = tl.program_id(2)

    context_len = tl.load(
        context_lengths_ptr + batch_id
    )

    cache_slot = tl.load(cache_slots_ptr + batch_id)

    qh_p_kv = hq // hkv

    split_start = tl.load(cum_splits_ptr + batch_id)
    local_split_start = split_id - split_start
    local_split_end = local_split_start + tl.arange(0, BLOCKS_PER_SPLIT)

    block_pos = BLOCK_TABLE_WIDTH * cache_slot + split_id * BLOCKS_PER_SPLIT

    '''blocks = tl.load(
        block_table_ptr + block_pos
    )'''

    kv_head_id = query_head_id // qh_p_kv

    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < d

    q_pos = (batch_id * hq + query_head_id) * d + head_offsets
    q_tile = tl.load(q_ptr + q_pos, mask=head_mask, other=0.0)

    partial_max = tl.cast(-float("inf"), tl.float32)
    partial_sum = tl.cast(0.0, tl.float32)
    partial_accum = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for kv_block_portion in tl.range(local_split_start * BLOCK_SIZE, BLOCKS_PER_SPLIT * local_split_start * BLOCK_SIZE, BLOCK_N):
        local_block_id = kv_block_portion // BLOCK_SIZE
        kv_block_mod = kv_block_portion % BLOCK_SIZE
        kv_offsets = kv_block_mod + tl.arange(0, BLOCK_N)
        kv_mask = kv_offsets < BLOCK_SIZE
        #block = blocks[local_block_id]

        block = tl.load(
            block_table_ptr
            + block_pos
            + local_block_id
        )

        remaining = tl.min(BLOCK_SIZE, context_len - local_block_id * BLOCK_SIZE)
        context_mask = kv_offsets < remaining
        kv_context_mask = kv_mask & context_mask

        kv_row_base = (block * BLOCK_SIZE + kv_offsets) * hkv + kv_head_id
        kv_pos = kv_row_base[:, None] + head_offsets[None, :]
        load_mask = kv_context_mask[:, None] & head_mask[None, :]

        k_tile = tl.load(
            k_cache_ptr + kv_pos,
            mask=load_mask,
            other=0.0
        )
        v_tile = tl.load(
            v_cache_ptr + kv_pos,
            mask=load_mask,
            other=0.0
        )

        s_tile = scale * tl.squeeze(tl.dot(k_tile, q_tile), dim = 1)
        s_tile = tl.where(kv_mask, s_tile, -float("inf"))
        tile_max = tl.max(s_tile, axis = -1)
        old_max = partial_max
        new_max = tl.maximum(tile_max, old_max)
        alpha = tl.exp(old_max - new_max)
        p = tl.exp(s_tile - new_max)
        partial_sum = alpha * partial_sum + tl.sum(p, axis=-1)
        partial_accum = tl.squeeze(alpha * partial_accum + tl.dot(p[None, :], v_tile.to(p.dtype)), dim = 0)

        partial_max = new_max

    pmax_pos = (batch_id * hq + query_head_id) * TOTAL_SPLITS + split_id
    psum_pos = pmax_pos
    tl.store(partial_max_ptr + pmax_pos, partial_max)
    tl.store(partial_sum_ptr + psum_pos, partial_sum)
    paccum_pos = ((batch_id * hq + query_head_id) * TOTAL_SPLITS + split_id) * d + head_offsets
    tl.store(partial_accum_ptr + paccum_pos, partial_accum)

@triton.jit
def decode_attention_reduction_kernel(
    pmax_ptr,
        psum_ptr,
        paccum_ptr,
        output_ptr,

        head_dim,
        hq,

        TOTAL_SPLITS: tl.constexpr,
        HEAD_DIM: tl.constexpr
):

    batch_id = tl.program_id(0)
    query_head_id = tl.program_id(1)

    split_offset = tl.arange(0, TOTAL_SPLITS)
    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < hq

    p_max_pos = (batch_id * hq + query_head_id) * TOTAL_SPLITS + split_offset
    p_maxes = tl.load(pmax_ptr + p_max_pos)
    global_max = tl.max(p_maxes, axis = 0)
    corrections = tl.exp(p_maxes - global_max)

    p_sum_pos = p_max_pos
    p_sums = tl.load(psum_ptr + p_sum_pos)
    denominator = tl.sum(corrections * p_sums, axis=-1)

    p_accum_pos = ((batch_id * hq + query_head_id) * TOTAL_SPLITS + split_offset[:, None]) * head_dim + head_offsets[
        None, :]
    p_accums = tl.load(paccum_ptr + p_accum_pos, mask=head_mask[None, :])
    numerator = tl.sum(
        corrections[:, None] * p_accums,
        axis=-2
    )

    output = numerator / denominator
    output_pos = (batch_id * hq + query_head_id) * head_dim + head_offsets
    tl.store(output_ptr + output_pos, output, mask=head_mask)

import math

import torch as t
import triton
import triton.language as tl


def decode_attention_wrapper(
    q,
    paged_kv_cache,
    cache_batch_context
):
    b, hq, d = q.shape

    k_cache = paged_kv_cache.k
    v_cache = paged_kv_cache.v

    assert k_cache.shape == v_cache.shape
    assert k_cache.ndim == 4

    _, cache_block_size, hkv, cache_d = k_cache.shape

    assert cache_d == d
    assert hq % hkv == 0
    assert cache_block_size == paged_kv_cache.block_size

    BLOCK_SIZE = paged_kv_cache.block_size
    BLOCKS_PER_SPLIT = 4

    assert BLOCK_SIZE > 0
    assert BLOCK_SIZE & (BLOCK_SIZE - 1) == 0

    HEAD_DIM = triton.next_power_of_2(d)
    SM_SCALE = 1.0 / math.sqrt(d)

    cache_slots = cache_batch_context.cache_slots
    context_lengths = cache_batch_context.context_lengths

    # Keep the FULL block table.
    # cache_slots selects the correct row inside the kernel.
    block_table = cache_batch_context.block_table
    block_table_width = block_table.shape[1]

    assert context_lengths.shape[0] == b
    assert cache_slots.shape[0] == b

    blocks_per_sequence = (
        context_lengths + BLOCK_SIZE - 1
    ) // BLOCK_SIZE

    splits_per_sequence = (
        blocks_per_sequence + BLOCKS_PER_SPLIT - 1
    ) // BLOCKS_PER_SPLIT

    splits_per_sequence = splits_per_sequence.to(t.long)

    cum_splits = t.cat([
        t.zeros(1, dtype=t.long, device=q.device),
        t.cumsum(splits_per_sequence, dim=0)
    ])

    # Needed as a Python integer for allocation / launch.
    total_splits = int(cum_splits[-1].item())

    output = t.empty(
        (b, hq, d),
        dtype=q.dtype,
        device=q.device
    )

    if total_splits == 0:
        output.zero_()
        return output

    # global split -> batch
    #
    # Long-term: I would build/cache this metadata when
    # cache_batch_context itself is constructed rather than
    # rebuilding it in the hot decode path.
    split_batch_ids = t.repeat_interleave(
        t.arange(b, device=q.device, dtype=t.long),
        splits_per_sequence,
        output_size=total_splits
    )

    partial_max = t.empty(
        (total_splits, hq),
        dtype=t.float32,
        device=q.device
    )

    partial_sum = t.empty(
        (total_splits, hq),
        dtype=t.float32,
        device=q.device
    )

    partial_accum = t.empty(
        (total_splits, hq, d),
        dtype=t.float32,
        device=q.device
    )

    grid = (total_splits, hq)

    decode_attention_split_kernel[grid](
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
        split_batch_ids,

        D=d,
        HQ=hq,
        HKV=hkv,
        SM_SCALE=SM_SCALE,

        BLOCK_SIZE=BLOCK_SIZE,
        BLOCKS_PER_SPLIT=BLOCKS_PER_SPLIT,
        BLOCK_TABLE_WIDTH=block_table_width,
        HEAD_DIM=HEAD_DIM,
    )

    grid = (b, hq)

    decode_attention_reduction_kernel[grid](
        partial_max,
        partial_sum,
        partial_accum,
        output,

        cum_splits,

        D=d,
        HQ=hq,
        HEAD_DIM=HEAD_DIM,
    )

    return output


@triton.jit
def decode_attention_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,

    partial_max_ptr,
    partial_sum_ptr,
    partial_accum_ptr,

    cache_slots_ptr,
    context_lengths_ptr,
    block_table_ptr,

    cum_splits_ptr,
    split_batch_ids_ptr,

    D: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    SM_SCALE: tl.constexpr,

    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_SPLIT: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    global_split_id = tl.program_id(0)
    query_head_id = tl.program_id(1)

    # Resolve global split -> sequence.
    batch_id = tl.load(
        split_batch_ids_ptr + global_split_id
    )

    sequence_split_start = tl.load(
        cum_splits_ptr + batch_id
    )

    local_split_id = (
        global_split_id - sequence_split_start
    )

    context_len = tl.load(
        context_lengths_ptr + batch_id
    )

    cache_slot = tl.load(
        cache_slots_ptr + batch_id
    )

    queries_per_kv_head = HQ // HKV
    kv_head_id = query_head_id // queries_per_kv_head

    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < D

    q_pos = (
        (batch_id * HQ + query_head_id) * D
        + head_offsets
    )

    q_tile = tl.load(
        q_ptr + q_pos,
        mask=head_mask,
        other=0.0
    ).to(tl.float32)

    # Scale Q once instead of every score tile.
    q_tile *= SM_SCALE

    partial_max = -float("inf")
    partial_sum = 0.0

    partial_accum = tl.zeros(
        (HEAD_DIM,),
        dtype=tl.float32
    )

    token_offsets = tl.arange(0, BLOCK_SIZE)

    # Each iteration handles ONE physical KV-cache block.
    #
    # This is important for a paged cache: adjacent logical
    # blocks are not guaranteed to be physically adjacent.
    for block_in_split in tl.static_range(
        0,
        BLOCKS_PER_SPLIT
    ):
        logical_block_id = (
            local_split_id * BLOCKS_PER_SPLIT
            + block_in_split
        )

        block_valid = (
            logical_block_id * BLOCK_SIZE < context_len
        ) & (
            logical_block_id < BLOCK_TABLE_WIDTH
        )

        physical_block = tl.load(
            block_table_ptr
            + cache_slot * BLOCK_TABLE_WIDTH
            + logical_block_id,
            mask=block_valid,
            other=0
        )

        absolute_token = (
            logical_block_id * BLOCK_SIZE
            + token_offsets
        )

        token_mask = (
            block_valid
            & (absolute_token < context_len)
        )

        # Cache layout assumed:
        #
        # [physical_block, token, kv_head, head_dim]
        #
        kv_pos = (
            (
                (
                    physical_block * BLOCK_SIZE
                    + token_offsets[:, None]
                ) * HKV
                + kv_head_id
            ) * D
            + head_offsets[None, :]
        )

        load_mask = (
            token_mask[:, None]
            & head_mask[None, :]
        )

        k_tile = tl.load(
            k_cache_ptr + kv_pos,
            mask=load_mask,
            other=0.0
        ).to(tl.float32)

        v_tile = tl.load(
            v_cache_ptr + kv_pos,
            mask=load_mask,
            other=0.0
        ).to(tl.float32)

        # [BLOCK_SIZE]
        #
        # Decode has one query, so a direct reduction is much
        # cleaner than forcing this into a 2D tl.dot.
        s_tile = tl.sum(
            k_tile * q_tile[None, :],
            axis=1
        )

        s_tile = tl.where(
            token_mask,
            s_tile,
            -float("inf")
        )

        tile_max = tl.max(
            s_tile,
            axis=0
        )

        new_max = tl.maximum(
            partial_max,
            tile_max
        )

        alpha = tl.exp(
            partial_max - new_max
        )

        p = tl.exp(
            s_tile - new_max
        )

        partial_sum = (
            alpha * partial_sum
            + tl.sum(p, axis=0)
        )

        partial_accum = (
            alpha * partial_accum
            + tl.sum(
                p[:, None] * v_tile,
                axis=0
            )
        )

        partial_max = new_max

    partial_pos = (
        global_split_id * HQ
        + query_head_id
    )

    tl.store(
        partial_max_ptr + partial_pos,
        partial_max
    )

    tl.store(
        partial_sum_ptr + partial_pos,
        partial_sum
    )

    partial_accum_pos = (
        partial_pos * D
        + head_offsets
    )

    tl.store(
        partial_accum_ptr + partial_accum_pos,
        partial_accum,
        mask=head_mask
    )


@triton.jit
def decode_attention_reduction_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_accum_ptr,
    output_ptr,

    cum_splits_ptr,

    D: tl.constexpr,
    HQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_head_id = tl.program_id(1)

    split_start = tl.load(
        cum_splits_ptr + batch_id
    )

    split_end = tl.load(
        cum_splits_ptr + batch_id + 1
    )

    # First pass: obtain the true max across this sequence's
    # partial softmaxes.
    global_max = -float("inf")

    for global_split_id in tl.range(
        split_start,
        split_end
    ):
        partial_pos = (
            global_split_id * HQ
            + query_head_id
        )

        split_max = tl.load(
            partial_max_ptr + partial_pos
        )

        global_max = tl.maximum(
            global_max,
            split_max
        )

    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < D

    numerator = tl.zeros(
        (HEAD_DIM,),
        dtype=tl.float32
    )

    denominator = 0.0

    # Second pass: correct every split into the global
    # softmax coordinate system.
    for global_split_id in tl.range(
        split_start,
        split_end
    ):
        partial_pos = (
            global_split_id * HQ
            + query_head_id
        )

        split_max = tl.load(
            partial_max_ptr + partial_pos
        )

        correction = tl.exp(
            split_max - global_max
        )

        split_sum = tl.load(
            partial_sum_ptr + partial_pos
        )

        denominator += (
            correction * split_sum
        )

        partial_accum_pos = (
            partial_pos * D
            + head_offsets
        )

        split_accum = tl.load(
            partial_accum_ptr
            + partial_accum_pos,
            mask=head_mask,
            other=0.0
        )

        numerator += (
            correction * split_accum
        )

    output = tl.where(
        denominator > 0.0,
        numerator / denominator,
        0.0
    )

    output_pos = (
        (batch_id * HQ + query_head_id) * D
        + head_offsets
    )

    tl.store(
        output_ptr + output_pos,
        output,
        mask=head_mask
    )