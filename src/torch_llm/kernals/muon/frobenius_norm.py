import triton
import triton.language as tl

@triton.jit()
def frobenius_norm_partial_sum_kernal(
    Bbuf_ptr,
    partial_sum_ptr,
    pid_to_mmtm_mat_ptr,
    length_cumsum_ptr,
    launches_per_mmtm_mat_ptr,
    ELEMENTS_PER_WORKER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    matnum = tl.load(pid_to_mmtm_mat_ptr + pid)

    prev_length = tl.load(length_cumsum_ptr + matnum).to(tl.int64)
    end_length = tl.load(length_cumsum_ptr + matnum + 1).to(tl.int64)
    base_launch = tl.load(launches_per_mmtm_mat_ptr + matnum)
    mat_launch = (pid - base_launch).to(tl.int64)

    partial_sum_tile = tl.full((), 0.0, tl.float32)  # local sum first

    for BLOCK_START in tl.range(0, ELEMENTS_PER_WORKER, BLOCK_SIZE):
        local_offsets = BLOCK_START + tl.arange(0, BLOCK_SIZE)
        block_offsets = prev_length + mat_launch * ELEMENTS_PER_WORKER + local_offsets
        block_mask = (block_offsets < end_length) & (local_offsets < ELEMENTS_PER_WORKER)

        mmtm_buf_tile = tl.load(Bbuf_ptr + block_offsets, mask=block_mask, other=0.0).to(tl.float32)
        partial_sum_tile += tl.sum(mmtm_buf_tile * mmtm_buf_tile, axis=0)

    tl.atomic_add(partial_sum_ptr + matnum, partial_sum_tile)  # workers share this sum


@triton.jit()
def frobenius_norm_normalize_kernal(
    Bbuf_ptr,
    sum_ptr,
    pid_to_mmtm_mat_ptr,
    length_cumsum_ptr,
    launches_per_mmtm_mat_ptr,
    eps,
    ELEMENTS_PER_WORKER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    matnum = tl.load(pid_to_mmtm_mat_ptr + pid)
    sum_tile = tl.load(sum_ptr + matnum).to(tl.float32)
    denominator = tl.where(sum_tile == 0.0, 1.0, tl.sqrt(sum_tile) + eps)

    prev_length = tl.load(length_cumsum_ptr + matnum).to(tl.int64)
    end_length = tl.load(length_cumsum_ptr + matnum + 1).to(tl.int64)
    base_launch = tl.load(launches_per_mmtm_mat_ptr + matnum)
    mat_launch = (pid - base_launch).to(tl.int64)

    for BLOCK_START in tl.range(0, ELEMENTS_PER_WORKER, BLOCK_SIZE):
        local_offsets = BLOCK_START + tl.arange(0, BLOCK_SIZE)
        block_offsets = prev_length + mat_launch * ELEMENTS_PER_WORKER + local_offsets
        block_mask = (block_offsets < end_length) & (local_offsets < ELEMENTS_PER_WORKER)

        mmtm_buf_tile = tl.load(Bbuf_ptr + block_offsets, mask=block_mask, other=0.0).to(tl.float32)
        mmtm_norm_buf_tile = tl.where(sum_tile == 0.0, 0.0, mmtm_buf_tile / denominator)
        tl.store(Bbuf_ptr + block_offsets, mmtm_norm_buf_tile, mask=block_mask)