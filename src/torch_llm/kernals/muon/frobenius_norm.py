import triton
import triton.language as tl
import math

@triton.jit()
def frobenius_norm_partial_sum_kernal(
        Bbuf_ptr,  # Bs buffer pointer
        partial_sum_ptr,  # writing to this

        pid_to_mmtm_mat_ptr,  # map pid to which parameter
        length_cumsum_ptr,  # determine offset mask cutoff and buffer load
        launches_per_mmtm_mat_ptr, #determine launches per mmtm to determine launch number with pid ie pid - [num] * elements per

        ELEMENTS_PER_WORKER: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    matnum = tl.load(pid_to_mmtm_mat_ptr + pid) #determine parameter matrix
    partial_sum_tile = tl.load(partial_sum_ptr + matnum)  # accumulate to

    prev_length = tl.load(length_cumsum_ptr + matnum) #length up to point
    end_length = tl.load(length_cumsum_ptr + matnum + 1) #end length for buffer

    base_launch = tl.load(launches_per_mmtm_mat_ptr + matnum)
    mat_launch = pid - base_launch

    for BLOCK_START in tl.range(0, ELEMENTS_PER_WORKER, BLOCK_SIZE):
        block_offsets = prev_length + mat_launch * ELEMENTS_PER_WORKER + tl.arange(0, BLOCK_START)
        block_mask = block_offsets <= end_length

        mmtm_buf_tile = tl.load(
            Bbuf_ptr + block_offsets,
            mask=block_mask,
            other=0.0
        )

        partial_sum_tile += tl.sum(mmtm_buf_tile * mmtm_buf_tile, dim=0)

    tl.store(partial_sum_ptr + matnum, partial_sum_tile)


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

    matnum = tl.load(pid_to_mmtm_mat_ptr + pid)  # determine parameter matrix
    sum_tile = tl.load(sum_ptr + matnum)  # accumulate to

    prev_length = tl.load(length_cumsum_ptr + matnum)  # length up to point
    end_length = tl.load(length_cumsum_ptr + matnum + 1)  # end length for buffer

    base_launch = tl.load(launches_per_mmtm_mat_ptr + matnum)
    mat_launch = pid - base_launch

    for BLOCK_START in tl.range(0, ELEMENTS_PER_WORKER, BLOCK_SIZE):
        block_offsets = prev_length + mat_launch * ELEMENTS_PER_WORKER + tl.arange(0, BLOCK_START)
        block_mask = block_offsets <= end_length

        mmtm_buf_tile = tl.load(
            Bbuf_ptr + block_offsets,
            mask=block_mask,
            other=0.0
        )

        mmtm_norm_buf_tile = tl.where(
            sum_tile != 0,
            mmtm_buf_tile / (math.sqrt(sum_tile) + eps),
            0.0
        )

        tl.store(Bbuf_ptr + block_offsets, mmtm_norm_buf_tile, mask=block_mask)



