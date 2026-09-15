import torch as t
import triton
import triton.language as tl
import math
from jaxtyping import Shaped
from muon.frobenius_norm import frobenius_norm_partial_sum_kernal, frobenius_norm_normalize_kernal

def muon_ns_kernal_wrapper(
        nesterov_mmtm_matrices_buffer: Shaped[t.Tensor, "tp"],
        orig_mmtm_matrices_metadata: Shaped[t.Tensor, "p 2"],
        ns_steps,
):
    '''
    Takes flat B matrices concatenated such that all parameters along 1 dim.
    Generally parameter matrices are taken from same parameter group.
    Works heterogeneously through masking and packing of parameters.
    Parameter shapes may be inputted transposed as to reduce intermediate NS dims.
    Preserves original parameter metadata to determine cuts, recreate shapes and output.
    Passes through 4 kernals:
    Frobenius normalization partial sum kernal.
    Frobenius normalization  kernal.
    A = XXT kernal.
    Y = AX kernal.
    Xnew = aX + bY + cAY kernal.
    As in Newton-Schulz algorithm, iterates through latter 3 kernals ns_steps.
    :param nesterov_mmtm_matrices_buffer:
    :param orig_param_metadata:
    :return:
    '''

    current_buffer = nesterov_mmtm_matrices_buffer

    #1 frobenius norm kernal pratial sum entry
    # create partial sums buffer
    # load global tiles into kernal
    # use metadata to determine which global tile it is to know where to write to
    # and what matrix to extract from
    #load partial sums of chunks (not reducing, simply loading a tile) into partial sum buffer. NOT SQUARE ROOTED.

    ELEMENTS_PER_WORKER = 102400 #total elements per worker
    BLOCK_SIZE = 1024 #number of elements summated per loop


    #extract required number of workers by summating total tiles required across all matrices

    workers_per_mmtm_mat = t.tensor([
        (rows * cols + ELEMENTS_PER_WORKER - 1) // ELEMENTS_PER_WORKER
        for rows, cols in orig_mmtm_matrices_metadata
    ])

    workers_req = sum(
        (rows * cols + ELEMENTS_PER_WORKER - 1) // ELEMENTS_PER_WORKER
        for rows, cols in orig_mmtm_matrices_metadata
    ) # dont want 7 million processes so larger block sizes


    # need to extract using which matrix found to determine cutoff mask
    mmtm_lengths = t.prod(orig_mmtm_matrices_metadata, dim=1)

    lengths_cumsum = tl.cat([
        t.zeros(
            1,
            device = current_buffer.device,
            dtype = current_buffer.dtype,
        ),
        mmtm_lengths.cumsum(dim=0)
    ])

    partial_sums = t.zeros((orig_mmtm_matrices_metadata.shape[0],), dtype=t.float32, device="cuda")


    #simple pid table. not extremely memory efficient, but compared to other memory sinks is negligible for now
    mmtm_matrices_arranged = t.arange(nesterov_mmtm_matrices_buffer.shape[0], dtype=t.float32, device="cuda")

    pid_to_mmtm_mat = t.repeat_interleave(
        mmtm_matrices_arranged,
        repeats=workers_per_mmtm_mat,
    )

    grid = (workers_req,)

    frobenius_norm_partial_sum_kernal[grid](
        current_buffer,
        partial_sums,

        pid_to_mmtm_mat,
        lengths_cumsum,
        workers_per_mmtm_mat,

        ELEMENTS_PER_WORKER,
        BLOCK_SIZE,
    )

    frobenius_norm_normalize_kernal[grid](
        current_buffer,
        partial_sums,

        pid_to_mmtm_mat,
        lengths_cumsum,
        workers_per_mmtm_mat,
        10e-8,

        ELEMENTS_PER_WORKER,
        BLOCK_SIZE,
    )

    #now current buffer is normalized

    #entering iteration stage

    a = 3.4445
    b = -4.7750
    c = 2.0315

    A = t.empty((
        sum(param.shape[0] ** 2 for param in orig_mmtm_matrices_metadata),
    ), dtype=t.float32, device=current_buffer.device) #buffer representation to make life easier, real shape per param is (M, M)

    Y = t.empty





    for _ in range(ns_steps):

        #1st kernal pass current_buffer, A ptr (M, M) when loading, for now needs to be empty like new construction

        BLOCK_M = 64  # output row
        BLOCK_N = 64  # output column
        BLOCK_K = 64  # reducing over XXT: (M, K) (K, M) -> (M, M)






    #unpack and untranspose




