import triton
import triton.lang as tl

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

        BLOCK_SIZE: tl.constexpr, # no more num splits because we are doing a split per logical block WAIT WHAT?
        BLOCK_N: tl.contsexpr,
        HEAD_DIM: tl.contsexpr,
):
    batch_id = tl.program_id(0)
    query_head_id = tl.program_id(1)
    split_id = tl.program_id(2)

    context_len = tl.load(
        context_lengths_ptr + batch_id
    )

    block_table = tl.load(
        block_table_ptr + batch_id)

    tps = (context_len + NUM_SPLITS - 1) // NUM_SPLITS  # SWSTICH AND FIX USING CUM SPLITS
    qh_p_kv = hq // hkv

    # NO!
    split_start = split_id * tps
    split_end = min(split_start + tps, seq_len)




    kv_head_id = query_head_id // qh_p_kv

    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < d

    q_pos = (batch_id * hq + query_head_id) * d
    q_tile = tl.load(q_ptr + q_pos, mask=head_mask, other=0.0)

    partial_max = tl.cast(-float("inf"), tl.float32)
    partial_sum = tl.cast(0.0, tl.float32)
    partial_accum = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for kv_start in tl.range(split_start, split_end, BLOCK_N):