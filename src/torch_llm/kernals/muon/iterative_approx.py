import triton
import triton.language as tl

@triton.jit()
def ns_x_xtrans_kernel(
        a_ptr,
        x_ptr,

        pid_to_group_ptr,
        length_cumsums_ptr,
        cum_group_programs_ptr,
        group_dims_ptr,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.load(pid_to_group_ptr + pid)

    start = tl.load(length_cumsums_ptr + group)
    end = tl.load(length_cumsums_ptr + group + 1)

    local_program = pid - tl.load(cum_group_programs_ptr + group)
    shape = tl.load(group_dims_ptr + group)