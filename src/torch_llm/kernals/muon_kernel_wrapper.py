import torch as t
import triton
import triton.language as tl
import math
from jaxtyping import Shaped
from torch import device

from kernals.muon.frobenius_norm import frobenius_norm_partial_sum_kernal, frobenius_norm_normalize_kernal
from torch_llm.kernals.muon.iterative_approx import ns_x_xtrans_kernel, ns_a_x_kernel, ns_a_y_kernel, \
    ns_x_resolve_kernel
import time

@t.no_grad()
def muon_ns_kernal_wrapper(
        nesterov_mmtm_matrices_buffer: Shaped[t.Tensor, "tp"],
        orig_mmtm_matrices_metadata: Shaped[t.Tensor, "p 2"],
        ns_steps,
) -> Shaped[t.Tensor, "tp"]:
    """Normalize and update packed original row-major matrices in place.

    Metadata has shape (num_matrices, 2), with original (rows, cols).
    Input must be a contiguous flat CUDA FP16, BF16, or FP32 tensor.
    Tall matrices are transposed for the iteration, then restored.
    Returns the input buffer, retaining its original layout and dtype.
    """
    current_buffer = nesterov_mmtm_matrices_buffer
    if not current_buffer.is_cuda or current_buffer.ndim != 1 or not current_buffer.is_contiguous():
        raise ValueError("expected a contiguous flat CUDA buffer")
    if current_buffer.dtype not in (t.float16, t.bfloat16, t.float32):
        raise ValueError("expected float16, bfloat16, or float32")
    if not isinstance(ns_steps, int) or ns_steps < 0:
        raise ValueError("ns_steps must be a nonnegative integer")
    if orig_mmtm_matrices_metadata.ndim != 2 or orig_mmtm_matrices_metadata.shape[1] != 2:
        raise ValueError("metadata must contain one (rows, cols) pair per matrix")
    if orig_mmtm_matrices_metadata.dtype not in (t.int32, t.int64):
        raise ValueError("metadata must use int32 or int64")

    # host shapes just for packing and restoring the tall matrices
    original_shapes = orig_mmtm_matrices_metadata.detach().cpu().tolist()
    if any(r < 0 or c < 0 for r, c in original_shapes):
        raise ValueError("matrix dimensions cannot be negative")

    if sum(r * c for r, c in original_shapes) != current_buffer.numel():
        raise ValueError("matrix lengths must add up to the buffer length")
    if current_buffer.numel() == 0:
        return current_buffer

    orig_mmtm_matrices_metadata = orig_mmtm_matrices_metadata.to(
        device=current_buffer.device,
        dtype=t.long,
    )
    rows = orig_mmtm_matrices_metadata[:, 0]
    cols = orig_mmtm_matrices_metadata[:, 1]

    M = t.minimum(rows, cols)
    K = t.maximum(rows, cols)

    # min/max changes shape, so tall matrices need an actual transpose
    start = 0
    for r, c in original_shapes:
        end = start + r * c
        if r > c and r * c:
            matrix = current_buffer[start:end].view(r, c)
            current_buffer[start:end].copy_(matrix.t().contiguous().view(-1))
        start = end

    # 1 frobenius norm partial sum entry. NOT SQUARE ROOTED.
    ELEMENTS_PER_WORKER = 102400  # total elements per worker
    BLOCK_SIZE = 1024  # number of elements summated per loop

    workers_per_mmtm_mat = (
        rows * cols + ELEMENTS_PER_WORKER - 1
    ) // ELEMENTS_PER_WORKER

    # counts build the pid table, cumulative counts give each matrix its first pid
    workers_cumsum = t.cat([
        t.zeros(1, device=current_buffer.device, dtype=t.long),
        workers_per_mmtm_mat.cumsum(dim=0),
    ])

    mmtm_lengths = t.prod(orig_mmtm_matrices_metadata, dim=1)
    lengths_cumsum = t.cat([
        t.zeros(1, device=current_buffer.device, dtype=t.long),
        mmtm_lengths.cumsum(dim=0),
    ])

    partial_sums = t.zeros(
        (orig_mmtm_matrices_metadata.shape[0],),
        dtype=t.float32,
        device=current_buffer.device,
    )

    # simple pid table, each matrix repeated by its worker count
    mmtm_matrices_arranged = t.arange(
        orig_mmtm_matrices_metadata.shape[0], dtype=t.int32, device=current_buffer.device
    )
    pid_to_mmtm_mat = t.repeat_interleave(
        mmtm_matrices_arranged,
        repeats=workers_per_mmtm_mat,
    )

    workers_req = pid_to_mmtm_mat.numel()
    grid = (workers_req,)

    # ------------------ BENCHMARK -------------------
    t.cuda.synchronize()
    t1 = time.perf_counter()
    # ------------------ BENCHMARK -------------------

    frobenius_norm_partial_sum_kernal[grid](
        current_buffer,
        partial_sums,

        pid_to_mmtm_mat,
        lengths_cumsum,
        workers_cumsum,

        ELEMENTS_PER_WORKER,
        BLOCK_SIZE,
    )

    # ------------------ BENCHMARK -------------------
    #t.cuda.synchronize()
    #t11 = time.perf_counter()
    #print(f"[frob partial sum kernel] {t11 - t1:.4f}s")

    #t.cuda.synchronize()
    #t2 = time.perf_counter()
    # ------------------ BENCHMARK -------------------
    frobenius_norm_normalize_kernal[grid](
        current_buffer,
        partial_sums,

        pid_to_mmtm_mat,
        lengths_cumsum,
        workers_cumsum,
        10e-8,

        ELEMENTS_PER_WORKER,
        BLOCK_SIZE,
    )
    ## ------------------ BENCHMARK -------------------
    #t.cuda.synchronize()
    #t22 = time.perf_counter()
    #print(f"[frob normalize kernel] {t22 - t2:.4f}s")
    # ------------------ BENCHMARK -------------------#

    # now current buffer is normalized, entering iteration stage
    a = 3.4445
    b = -4.7750
    c = 2.0315

    x_lengths = M * K
    a_lengths = M * M

    A = t.empty(
        int(a_lengths.sum().item()),
        dtype=current_buffer.dtype,  # dot operands need matching types
        device=current_buffer.device,
    )
    Y = t.empty_like(current_buffer)
    Z = t.empty_like(current_buffer)

    # ------------------ BENCHMARK -------------------
    #t.cuda.synchronize()
    #t7 = time.perf_counter()
    # ------------------ BENCHMARK -------------------

    for _ in range(ns_steps):
        # 1st kernal: XXT, output is (M, M)
        BLOCK_M = 64  # output row
        BLOCK_N = 64  # output column
        BLOCK_K = 64  # reducing over K

        programs_per_group = (
            ((M + BLOCK_M - 1) // BLOCK_M)
            * ((M + BLOCK_N - 1) // BLOCK_N)
        )

        cum_group_programs = t.cat([
            t.zeros(1, dtype=t.long, device=current_buffer.device),
            programs_per_group.cumsum(dim=0),
        ])

        groups_arranged = t.arange(
            orig_mmtm_matrices_metadata.shape[0], dtype=t.long, device=current_buffer.device
        )
        pid_to_group = t.repeat_interleave(
            groups_arranged,
            repeats=programs_per_group,
        )

        num_workers = pid_to_group.numel()  # python int for the grid
        grid = (num_workers,)

        A_group_dims = a_lengths
        A_lengths_cumsum = t.cat([
            t.zeros(1, device=current_buffer.device, dtype=t.long),
            A_group_dims.cumsum(dim=0),
        ])

        buffer_dimensions = t.stack((M, K), dim=1).contiguous()  # shape pairs, not lengths
        buffer_lengths_cumsum = t.cat([
            t.zeros(1, device=current_buffer.device, dtype=t.long),
            x_lengths.cumsum(dim=0),
        ])

        # ------------------ BENCHMARK -------------------
        #t.cuda.synchronize()
        #t3 = time.perf_counter()
        # ------------------ BENCHMARK -------------------

        ns_x_xtrans_kernel[grid](
            A,
            current_buffer,

            pid_to_group,
            A_lengths_cumsum,
            cum_group_programs,
            A_group_dims,
            buffer_lengths_cumsum,
            buffer_dimensions,

            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )

       # ------------------ BENCHMARK -------------------
        #t.cuda.synchronize()
        #t33 = time.perf_counter()
        #print(f"[ns x xtrans kernel] {t33 - t3:.4f}s")
        # ------------------ BENCHMARK -------------------

        # next two kernals output (M, K), so rebuild the pid table
        programs_per_group = (
            ((M + BLOCK_M - 1) // BLOCK_M)
            * ((K + BLOCK_N - 1) // BLOCK_N)
        )

        cum_group_programs = t.cat([
            t.zeros(1, dtype=t.long, device=current_buffer.device),
            programs_per_group.cumsum(dim=0),
        ])

        groups_arranged = t.arange(
            orig_mmtm_matrices_metadata.shape[0], dtype=t.long, device=current_buffer.device
        )
        pid_to_group = t.repeat_interleave(
            groups_arranged,
            repeats=programs_per_group,
        )

        num_workers = pid_to_group.numel()
        grid = (num_workers,)

        yb_dimensions = buffer_dimensions
        yb_lengths_cumsum = buffer_lengths_cumsum  # X, Y and Z share shapes and offsets

        # ------------------ BENCHMARK -------------------
        t.cuda.synchronize()
        t4 = time.perf_counter()
        # ------------------ BENCHMARK -------------------

        ns_a_x_kernel[grid](
            A,
            current_buffer,
            Y,

            pid_to_group,
            A_lengths_cumsum,
            cum_group_programs,
            A_group_dims,
            yb_lengths_cumsum,
            yb_dimensions,

            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )

        # ------------------ BENCHMARK -------------------
        #t.cuda.synchronize()
        #t44 = time.perf_counter()
        #print(f"[ns a x kernel] {t44 - t4:.4f}s")

        #t.cuda.synchronize()
        #t5 = time.perf_counter()
        # ------------------ BENCHMARK -------------------

        ns_a_y_kernel[grid](
            A,
            Y,
            Z,

            pid_to_group,
            A_lengths_cumsum,
            cum_group_programs,
            A_group_dims,
            yb_lengths_cumsum,
            yb_dimensions,

            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )

        # ------------------ BENCHMARK -------------------
        #t.cuda.synchronize()
        #t55 = time.perf_counter()
        #print(f"[ns a y kernel] {t55 - t5:.4f}s")

        #t.cuda.synchronize()
        #t6 = time.perf_counter()
        # ------------------ BENCHMARK -------------------

        BLOCK_SIZE = 512  # may need tuning and benchmarking
        end_length = current_buffer.numel()  # value, not a scalar tensor pointer
        grid = (triton.cdiv(end_length, BLOCK_SIZE),)

        ns_x_resolve_kernel[grid](
            current_buffer,
            Y,
            Z,

            a,
            b,
            c,
            end_length,

            BLOCK_SIZE,
        )

        # ------------------ BENCHMARK -------------------
        #t.cuda.synchronize()
        #t66 = time.perf_counter()
        #print(f"[ns x resolve kernel] {t66 - t6:.4f}s")

        # ------------------ BENCHMARK -------------------

    # ------------------ BENCHMARK -------------------
    #t.cuda.synchronize()
    #t77 = time.perf_counter()
    #print(f"[ns_steps all iterations] {t77 - t7:.4f}s")
    # ------------------ BENCHMARK -------------------

    start = 0
    for r, c in original_shapes:
        end = start + r * c
        if r > c and r * c:
            matrix = current_buffer[start:end].view(c, r)
            current_buffer[start:end].copy_(matrix.t().contiguous().view(-1))
        start = end

    return current_buffer





