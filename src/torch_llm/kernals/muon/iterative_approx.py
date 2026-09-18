import triton
import triton.language as tl

@triton.jit()
def ns_x_xtrans_kernel(
        a_ptr,
        x_ptr,

        pid_to_group_ptr,
        A_length_cumsums_ptr,
        cum_group_programs_ptr,
        A_group_dims_ptr,
        X_length_cumsums_ptr,
        X_dims_ptr,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.load(pid_to_group_ptr + pid)

    A_start = tl.load(A_length_cumsums_ptr + group)
    A_end = tl.load(A_length_cumsums_ptr + group + 1)

    local_program = pid - tl.load(cum_group_programs_ptr + group)
    A_shape = tl.sqrt(tl.load(A_group_dims_ptr + group))

    tiles_N = tl.ceil(A_shape, BLOCK_N)

    A_col_start = local_program % tiles_N #tile indices
    A_row_start = local_program // tiles_N #tile indices


    col_offsets = A_col_start * BLOCK_N + tl.arange(0, BLOCK_N)
    row_offsets = A_row_start * BLOCK_M + tl.arange(0, BLOCK_M)

    A_pos = (
        A_start
        + row_offsets[:, None] * A_shape
        + col_offsets[None, :]
    )

    A_col_mask = col_offsets < A_shape
    A_row_mask = row_offsets < A_shape

    A_store_mask = A_col_mask[None, :] & A_row_mask[:, None]

    A_tile_value = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_start = tl.load(X_length_cumsums_ptr + group)
    x_end = tl.load(X_length_cumsums_ptr + group + 1)

    x_height = tl.load(X_dims_ptr + 2 * group)
    x_width = tl.load(X_dims_ptr + 2 * group + 1)

    for K_START in range(0, x_width, BLOCK_K):
        k_offsets = K_START + tl.arange(0, BLOCK_K)

        x_row_positions = (
            x_start
            + row_offsets[:, None] * x_width
            + k_offsets[None, :]
        )

        x_col_positions = (
            x_start
            + col_offsets[:, None] * x_width
            + k_offsets[None, :]
        )

        x_row_mask = (
            (row_offsets[:, None] < x_height)
            & (k_offsets[:, None] < x_width)
        )

        x_col_mask = (
            (col_offsets[:, None] < x_height)
            & (k_offsets[None, :] < x_width)
        )

        x = tl.load(
            x_ptr + x_row_positions,
            mask=x_col_mask,
            other=0.0,
        )

        x_trans = tl.load(
            x_ptr + x_col_positions,
            mask=x_col_mask,
            other=0.0,
        )

        A_tile_value += tl.dot(x, x_trans)

    tl.store(
        a_ptr + A_pos,
        A_tile_value,
        mask=A_store_mask
    )


@triton.jit()
def ns_a_x_kernel(
        a_ptr,
        x_ptr,
        y_ptr,

        pid_to_group_ptr,
        A_length_cumsums_ptr,
        cum_group_programs_ptr,
        A_group_dims_ptr,
        X_length_cumsums_ptr, #Y and X share the same shape : M, K, whereas A is M, M
        X_dims_ptr,

        BLOCK_M: tl.constexpr, #output row into Y M
        BLOCK_N: tl.constexpr, #output column into Y K
        BLOCK_K: tl.constexpr, #reducing over (M, M) inner dimension
):

    pid = tl.program_id(0)
    group = tl.load(pid_to_group_ptr + pid)

    A_start = tl.load(A_length_cumsums_ptr + group)
    A_end = tl.load(A_length_cumsums_ptr + group + 1)

    local_program = pid - tl.load(cum_group_programs_ptr + group)
    A_shape = tl.sqrt(tl.load(A_group_dims_ptr + group))

    yx_start = tl.load(X_length_cumsums_ptr + group)
    yx_end = tl.load(X_length_cumsums_ptr + group + 1)

    yx_height = tl.load(X_dims_ptr + 2 * group)
    yx_width = tl.load(X_dims_ptr + 2 * group + 1)

    tiles_N = tl.ceil(yx_width, BLOCK_N)

    y_row_start = local_program // tiles_N
    y_col_start = local_program % tiles_N

    row_offsets = y_row_start * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = y_col_start * BLOCK_N + tl.arange(0, BLOCK_N)

    y_pos = (
        yx_start
        + row_offsets[:, None] * yx_width
        + col_offsets[None, :]
    )

    y_row_mask = row_offsets < yx_height
    y_col_mask = col_offsets < yx_width

    y_mask = y_row_mask[:, None] & y_col_mask[None, :]

    y_tile_vals = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for K_START in range(0, yx_height, BLOCK_K):
        k_offsets = K_START + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < yx_height

        A_pos = (
            A_start
            + row_offsets[:, None] * A_shape
            + k_offsets[None, :]
        )

        x_pos = (
            yx_start
            + k_offsets[:, None] * yx_width
            + col_offsets[None, :]
        )

        A_mask = k_mask[:, None] & y_row_mask[None, :] # looks counterintuitive but i was lazy so i reused

        x_mask = k_mask[:, None] & y_col_mask[None, :]

        x_tile = tl.load(
            x_ptr + x_pos,
            mask=x_mask,
            other=0.0,
        )

        A_tile = tl.load(
            a_ptr + A_pos,
            mask=A_mask,
            other=0.0,
        )

        y_tile_vals += tl.dot(A_tile, x_tile)

    tl.store(
        y_ptr + y_pos,
        y_tile_vals,
        mask=y_mask,
    )

@triton.jit()
def ns_a_y_kernel(
        a_ptr,
        y_ptr,
        z_ptr,

        pid_to_group_ptr,
        A_length_cumsums_ptr,
        cum_group_programs_ptr,
        A_group_dims_ptr,
        Y_length_cumsums_ptr,  # Y and Z share the same shape : M, K, whereas A is M, M
        Y_dims_ptr,

        BLOCK_M: tl.constexpr,  # output row into Z M
        BLOCK_N: tl.constexpr,  # output column into Z K
        BLOCK_K: tl.constexpr,  # reducing over (M, M) inner dimension
):
    pid = tl.program_id(0)
    group = tl.load(pid_to_group_ptr + pid)

    A_start = tl.load(A_length_cumsums_ptr + group)
    A_end = tl.load(A_length_cumsums_ptr + group + 1)

    local_program = pid - tl.load(cum_group_programs_ptr + group)
    A_shape = tl.sqrt(tl.load(A_group_dims_ptr + group))

    z_start = tl.load(Y_length_cumsums_ptr + group)
    z_end = tl.load(Y_length_cumsums_ptr + group + 1)

    z_height = tl.load(Y_dims_ptr + 2 * group)
    z_width = tl.load(Y_dims_ptr + 2 * group + 1)

    tiles_N = tl.ceil(z_width, BLOCK_N)

    z_row_start = local_program // tiles_N
    z_col_start = local_program % tiles_N

    row_offsets = z_row_start * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = z_col_start * BLOCK_N + tl.arange(0, BLOCK_N)

    z_pos = (
            z_start
            + row_offsets[:, None] * z_width
            + col_offsets[None, :]
    )

    z_row_mask = row_offsets < z_height
    z_col_mask = col_offsets < z_width

    z_mask = z_row_mask[:, None] & z_col_mask[None, :]

    z_tile_vals = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for K_START in range(0, z_height, BLOCK_K):
        k_offsets = K_START + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < z_height

        A_pos = (
                A_start
                + row_offsets[:, None] * A_shape
                + k_offsets[None, :]
        )

        y_pos = (
                z_start
                + k_offsets[:, None] * z_width
                + col_offsets[None, :]
        )

        A_mask = k_mask[:, None] & z_row_mask[None, :]  # looks counterintuitive but i was lazy so i reused

        y_mask = k_mask[:, None] & z_col_mask[None, :]

        y_tile = tl.load(
            y_ptr + y_pos,
            mask=y_mask,
            other=0.0,
        )

        A_tile = tl.load(
            a_ptr + A_pos,
            mask=A_mask,
            other=0.0,
        )

        z_tile_vals += tl.dot(A_tile, y_tile)

    tl.store(
        z_ptr + z_pos,
        z_tile_vals,
        mask=z_mask,
    )

@triton.jit()
def ns_x_resolve_kernel(
        x_ptr,
        y_ptr,
        z_ptr,

        a,
        b,
        c,
        end_length,

        BLOCK_SIZE: tl.constexpr, # simply iterating through in chunks because all are same dims
):
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE

    block_offsets = block + tl.arange(0, BLOCK_SIZE)
    block_mask = block_offsets < end_length

    x_tile = tl.load(x_ptr + block_offsets, mask=block_mask, other=0.0)
    y_tile = tl.load(y_ptr + block_offsets, mask=block_mask, other=0.0)
    z_tile = tl.load(z_ptr + block_offsets, mask=block_mask, other=0.0)

    tl.store(
        x_ptr + block_offsets,
        x_tile * a + y_tile * b + z_tile * c,
        mask=block_mask,
    )
