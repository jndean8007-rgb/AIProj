import torch as t
import triton
import triton.language as tl
from torch_llm.kernals.moe.scheduling import find_expert_for_tile_helper
from jaxtyping import Shaped

def grouped_down_input_grad(
        grad_output,
        down_weight,
        expert_offsets,
) -> Shaped[t.Tensor, 'A F']:

    A, D = grad_output.shape
    num_experts, D_down, F = down_weight.shape
    assert D_down == D
    assert expert_offsets.shape[0] == num_experts + 1

    BLOCK_M = 64 # number of expert token rows handled pp
    BLOCK_N = 64 # number of d_f features handled pp
    BLOCK_K = 64 # number of d_model features per chunk within program looped

    expert_counts = (
        expert_offsets[1:]
        - expert_offsets[:-1]
    )

    tiles_m_per_expert = (
        expert_counts + BLOCK_M - 1
    ) // BLOCK_M

    tiles_n = triton.cdiv(F, BLOCK_N)

    tiles_per_expert = (
        tiles_m_per_expert * tiles_n
    )

    expert_tile_offsets = t.cat([
        t.zeros(
            1,
            device=grad_output.device,
            dtype=expert_offsets.dtype
        ),
        t.cumsum(tiles_per_expert, dim=0)
    ])

    total_tiles = int(
        expert_tile_offsets[-1].item()
    )

    if total_tiles == 0:
        return t.zeros((A, F), dtype = grad_output.dtype, device = grad_output.device)

    desired_persistent_ctas = 36
    num_workers = min(desired_persistent_ctas, total_tiles)
    grid = (num_workers,)

    SEARCH_STEPS = num_experts.bit_length() + 1

    dh = t.empty(
        (A,F),
        device=grad_output.device,
        dtype=grad_output.dtype
    )

    grouped_down_input_grad_kernal[grid](
        grad_output,
        down_weight,
        expert_offsets,
        expert_tile_offsets,
        dh,

        F,
        D,
        total_tiles,

        SEARCH_STEPS,
        num_experts,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return dh




@triton.jit()
def grouped_down_input_grad_kernal(
        grad_output_ptr,
        down_weight_ptr,
        expert_offsets_ptr,
        expert_tile_offsets_ptr,
        dh_ptr,

        F,
        D,
        total_tiles,

        SEARCH_STEPS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_workers = tl.num_programs(0)

    tile_id = pid

    while tile_id < total_tiles:

        e = find_expert_for_tile_helper(
            expert_tile_offsets_ptr,
            tile_id,

            NUM_EXPERTS,
            SEARCH_STEPS
        )

        expert_offset = tl.load(expert_offsets_ptr + e)
        next_expert_offset = tl.load(expert_offsets_ptr + e + 1)

        expert_tile_offset = tl.load(expert_tile_offsets_ptr + e)

        local_tile_id = tile_id - expert_tile_offset

        tile_N = tl.cdiv(F, BLOCK_N)

        tile_m = local_tile_id // tile_N
        tile_n = local_tile_id % tile_N

        m_offsets = (
            expert_offset
            + tile_m * BLOCK_M
            + tl.arange(0, BLOCK_M)
        )

        n_offsets = (
            tile_n * BLOCK_N
            + tl.arange(0, BLOCK_N)
        )

        m_mask = m_offsets < next_expert_offset
        n_mask = n_offsets < F

        dh_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

        for k_start in tl.range(0, D, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < D

            down_weight_positions  = (
                e * F * D
                + k_offsets[:, None] * F
                + n_offsets[None, :]
            )

            down_tile = tl.load(
                down_weight_ptr + down_weight_positions,
                mask = n_mask[None, :] & k_mask[:, None]
            )

            grad_output_positions = (
                m_offsets[:, None] * D
                + k_offsets[None, :]
            )

            grad_output_tile = tl.load(
                grad_output_ptr + grad_output_positions,
                mask = m_mask[:, None] & k_mask[None, :]
            )

            dh_accum += tl.dot(grad_output_tile, down_tile)


        dh_positions = (
                m_offsets[:, None] * F
                + n_offsets[None, :]
        )

        tl.store(dh_ptr + dh_positions,
                dh_accum,
                mask = m_mask[:, None] & n_mask[None, :]
        )

        tile_id += num_workers

def grouped_down_weight_grad(
        grad_output,
        hidden,
        expert_offsets
):
    A, D = grad_output.shape
    hidden_A, F = hidden.shape
    assert hidden_A == A
    num_experts = expert_offsets.shape[-1] - 1

    BLOCK_M = 64 # dmodel rows of dW, 1 pp
    BLOCK_N = 64  # dff columns of dW, 1 pp
    BLOCK_K = 64 # assignments routed to expert, processed in chunks within program

    expert_counts = (
            expert_offsets[1:]
            - expert_offsets[:-1]
    )

    tiles_m = triton.cdiv(D, BLOCK_M)

    tiles_n = triton.cdiv(F, BLOCK_N)

    tiles_per_expert = (
            tiles_m * tiles_n
    )

    total_tiles = tiles_per_expert * num_experts

    if total_tiles == 0:
        return t.zeros((D, F), dtype=grad_output.dtype, device=grad_output.device)

    desired_persistent_ctas = 36
    num_workers = min(desired_persistent_ctas, total_tiles)
    grid = (num_workers,)


    dw = t.zeros(
        (num_experts, D, F),
        device=grad_output.device,
        dtype=grad_output.dtype
    )

    grouped_down_weight_grad_kernal[grid](
        grad_output,
        hidden,
        expert_offsets,
        dw,

        D,
        F,
        total_tiles,

        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return dw


@triton.jit()
def grouped_down_weight_grad_kernal(
        grad_output_ptr,
        hidden_ptr,
        expert_offsets_ptr,
        dw_ptr,

        D,
        F,
        total_tiles,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_workers = tl.num_programs(0)

    tile_id = pid

    while tile_id < total_tiles:
        tiles_M = tl.cdiv(D, BLOCK_M)
        tiles_N = tl.cdiv(F, BLOCK_N)

        tiles_per_expert = tiles_M * tiles_N

        e = tile_id // tiles_per_expert
        local_tile = tile_id % tiles_per_expert

        tile_m = local_tile // tiles_N
        tile_n = local_tile % tiles_N

        dw_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

        d_offsets = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        f_offsets = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

        d_mask = d_offsets < D
        f_mask = f_offsets < F

        expert_start = tl.load(expert_offsets_ptr + e)
        expert_end = tl.load(expert_offsets_ptr + e + 1)
        N_e = expert_end - expert_start

        #grad down weight E D F
        grad_down_positions = (
            e * D * F
            + d_offsets[:, None] * F
            + f_offsets[None, :]
        )

        for k_start in tl.range(0, N_e, BLOCK_K): #loads grad output and hidden and accums
            #loading into blockm, blockn therefore want gradout blockm, blockk @ hidden blockk blockn
            local_k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = local_k_offsets < N_e
            global_k_offsets = expert_start + local_k_offsets

            #grad output A, D
            #hidden A, F

            grad_output_T_positions = (
                    global_k_offsets[None, :] * D
                    + d_offsets[:, None]
            )#this one is transposed because we want blockm blockk. # logical transpose [D, K], without tl.trans

            grad_output_T_tile = tl.load(
                grad_output_ptr + grad_output_T_positions,
                mask = d_mask[:, None] & k_mask[None, :],
                other = 0.0
            )

            hidden_positions = (
                global_k_offsets[:, None] * F
                + f_offsets[None, :]
            ) # this one is fine because we want blockk blockn

            hidden_tile = tl.load(
                hidden_ptr + hidden_positions,
                mask = k_mask[:, None] & f_mask[None, :],
                other = 0.0
            )

            dw_accum += tl.dot(grad_output_T_tile, hidden_tile)

        tl.store(dw_ptr + grad_down_positions,
                 dw_accum,
                 mask=d_mask[:, None] & f_mask[None, :]
        )

        tile_id += num_workers





    #output is D, F for every expert
    # one block owns 1 expert, one BLOCKM chunk of D
    #one block N chunk of F
    # reduces against jagged dimensions K = Ne