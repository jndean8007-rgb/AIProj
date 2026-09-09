import torch as t
import triton
import triton.language as tl

from torch_llm.kernals.moe.scheduling import find_expert_for_tile_helper
from jaxtyping import Shaped


def grouped_swiglu_backward(
        grad_hidden, #A, F
        gate_weights, #E, F, D --- remember following linear semantics - row 'dots with' feature row
        up_weights, #E, F, D
        expert_inputs, #A, D
        expert_offsets, #E+1
):

    #kernal 1 launch for dg, du
    assert gate_weights.shape == up_weights.shape
    A, F = grad_hidden.shape
    D = expert_inputs.shape[-1]
    num_experts = expert_offsets.shape[-1] - 1
    dg = t.zeros((A, F), dtype=grad_hidden.dtype, device=grad_hidden.device)
    du = t.zeros((A, F), dtype=grad_hidden.dtype, device=grad_hidden.device)

    BLOCK_M = 64  # number of expert-token rows handled by one program
    BLOCK_N = 64  # number of d_ff features read from grad hidden at once and written to d
    BLOCK_K = 32  # number of d_model/input features handled per chunk in program (read from x)

    expert_counts = (
            expert_offsets[1:]
            - expert_offsets[:-1]
    )
    # Number of row tiles needed independently for each expert.
    tiles_m_per_expert = (
                 expert_counts + BLOCK_M - 1
         ) // BLOCK_M

    # Every expert has the same d_ff width.
    tiles_n = triton.cdiv(F, BLOCK_N)

    tiles_per_expert = (
            tiles_m_per_expert * tiles_n
    )

    expert_tile_offsets = t.cat([
        t.zeros(
            1,
            device=expert_inputs.device,
            dtype=tiles_per_expert.dtype,
        ),
        t.cumsum(tiles_per_expert, dim=0),
    ])

    # Total number of logical SwiGLU output tiles.
    total_tiles = int(
        expert_tile_offsets[-1].item()
    )

    #if total_tiles == 0:
        #return

    desired_persistent_ctas = 36  # i have 36 SMs on my gpu, so 1 persistent cpa per sm. may not be optimal.
    num_workers = min(total_tiles, desired_persistent_ctas)
    grid = (num_workers,)

    SEARCH_STEPS = num_experts.bit_length() + 1

    grouped_swiglu_recompute_grad_kernal[grid](
        expert_inputs,
        gate_weights,
        up_weights,
        grad_hidden,
        dg,
        du,
        expert_offsets,
        expert_tile_offsets,

        D,
        F,
        total_tiles,

        SEARCH_STEPS,
        num_experts,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    dx = t.empty(
        (A, D),
        device=expert_inputs.device,
        dtype=expert_inputs.dtype
    )

    #M is tokens pp
    #N is dmodel feature output
    #K is dff reduction dimension

    tiles_m_per_expert = (
        expert_counts + BLOCK_M - 1
    ) // BLOCK_M

    tiles_n = triton.cdiv(D, BLOCK_N)

    tiles_per_expert = (
            tiles_m_per_expert * tiles_n
    )

    expert_tile_offsets = t.cat([
        t.zeros(
            1,
            device=expert_inputs.device,
            dtype=tiles_per_expert.dtype,
        ),
        t.cumsum(tiles_per_expert, dim=0),
    ])

    # Total number of logical SwiGLU output tiles.
    total_tiles = int(
        expert_tile_offsets[-1].item()
    )


    desired_persistent_ctas = 36
    num_workers = min(desired_persistent_ctas, total_tiles)
    grid = (num_workers,)

    grouped_swiglu_input_grad[grid](
        dg,
        du,
        gate_weights,
        up_weights,
        dx,
        expert_offsets,
        expert_tile_offsets,

        D,
        F,
        total_tiles,

        SEARCH_STEPS,
        num_experts,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        )

    dw_gate = t.empty((num_experts, F, D), device=grad_hidden.device, dtype=grad_hidden.dtype)
    dw_up = t.empty((num_experts, F, D), device=grad_hidden.device, dtype=grad_hidden.dtype)

    #M is F, output rows pp
    #N is D, output columns pp
    #K is N_e ragged expert assignments within program

    tiles_m = triton.cdiv(F, BLOCK_M)
    tiles_n = triton.cdiv(D, BLOCK_N)

    tiles_per_expert = (
            tiles_m * tiles_n
    )

    total_tiles = tiles_per_expert * num_experts

    desired_persistent_ctas = 36
    num_workers = min(desired_persistent_ctas, total_tiles)
    grid = (num_workers,)

    SEARCH_STEPS = num_experts.bit_length() + 1

    grouped_swiglu_weight_grad[grid](
        expert_inputs,
        dg,
        du,
        dw_gate,
        dw_up,
        expert_offsets,

        D,
        F,
        total_tiles,

        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )

    return dx, dw_gate, dw_up,


@triton.jit()
def grouped_swiglu_recompute_grad_kernal(
        x_ptr,
        gate_weights_ptr,
        up_weights_ptr,
        dh_ptr,
        dg_ptr,
        du_ptr,
        expert_offsets_ptr,
        expert_tile_offsets_ptr,

        D,
        F,
        total_tiles,

        SEARCH_STEPS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_workers = tl.num_programs(0)

    tile_id = pid
    while tile_id < total_tiles:
        e = find_expert_for_tile_helper(
            expert_tile_offsets_ptr,
            tile_id,

            NUM_EXPERTS,
            SEARCH_STEPS,
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

        g_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        u_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, D, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < D

            x_positions = (
                m_offsets[:, None] * D
                + k_offsets[None, :]
            )

            x_tile = tl.load(
                x_ptr + x_positions,
                mask = m_mask[:, None] & k_mask[None, :],
                other = 0.0
            )

            w_gate_positions = (  # num experts, dff, dmodel #goofy standard
                e * D * F
                + n_offsets[None, :] * D
                + k_offsets[:, None]
            )

            w_gate_tile = tl.load(
                gate_weights_ptr + w_gate_positions,
                mask = k_mask[:, None] & n_mask[None, :],
                other=0.0
            )

            w_up_weight_positions = w_gate_positions

            w_up_tile = tl.load(
                up_weights_ptr + w_up_weight_positions,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            g_accum += tl.dot(x_tile, w_gate_tile)
            u_accum += tl.dot(x_tile, w_up_tile)

        gate = g_accum
        up = u_accum

        dh_positions = (
            m_offsets[:, None] * F
            + n_offsets[None, :]
        )

        dh_tile = tl.load(
            dh_ptr + dh_positions,
            mask = m_mask[:, None] & n_mask[None, :],
            other = 0.0
        )

        gate_sig = tl.sigmoid(gate)
        du = dh_tile * (gate * gate_sig)
        dg = dh_tile * up * (gate_sig * (1 + gate * (1 - gate_sig)))

        dg_positions = dh_positions
        du_positions = dh_positions

        tl.store(
            dg_ptr + dg_positions,
            dg,
            mask=m_mask[:, None] & n_mask[None, :]
        )

        tl.store(
            du_ptr + du_positions,
            du,
            mask=m_mask[:, None] & n_mask[None, :]
        )

        tile_id += num_workers


#Then du and dg feed into two different backward branches. (consider these like a throughput/helper)
#input gradiant:
@triton.jit()
def grouped_swiglu_input_grad(
        dg_ptr,
        du_ptr,
        gate_weights_ptr,
        up_weights_ptr,
        dx_ptr,
        expert_offsets_ptr,
        expert_tile_offsets_ptr,

        D,
        F,
        total_tiles,

        SEARCH_STEPS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
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

        tile_N = tl.cdiv(D, BLOCK_N)

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
        n_mask = n_offsets < D

        dx_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, F, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < F

            dg_positions = (
                m_offsets[:, None] * F
                + k_offsets[None, :]
            )

            du_positions = dg_positions

            dg_tile = tl.load(
                dg_ptr + dg_positions,
                mask = m_mask[:, None] & k_mask[None, :],
                other=0.0
            )

            du_tile = tl.load(
                du_ptr + du_positions,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0
            )

            wg_positions = (
                e * F * D #actual parameter layout rather than linear semantic
                + k_offsets[:, None] * D
                + n_offsets[None, :]
            )#E, F, D

            wu_positions = wg_positions

            wg_tile = tl.load(
                gate_weights_ptr + wg_positions,
                mask = k_mask[:, None] & n_mask[None, :],
                other = 0.0
            )

            wu_tile = tl.load(
                up_weights_ptr + wu_positions,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0
            )

            dx_accum += tl.dot(dg_tile, wg_tile) + tl.dot(du_tile, wu_tile)

        dx_positions = (
            m_offsets[:, None] * D
            + n_offsets[None, :]
        )

        tl.store(
            dx_ptr + dx_positions,
            dx_accum,
            mask=m_mask[:, None] & n_mask[None, :],
        )

        tile_id += num_workers



@triton.jit()
def grouped_swiglu_weight_grad(
        x_ptr,
        dg_ptr,
        du_ptr,
        dwgate_ptr,
        dwup_ptr,
        expert_offsets_ptr,

        D,
        F,
        total_tiles,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr
):
    #M is F, output row
    #N is D, output col
    #K is N_e, looped thru

    pid = tl.program_id(0)
    num_workers = tl.num_programs(0)

    tile_id = pid

    while tile_id < total_tiles:
        tiles_M = tl.cdiv(F, BLOCK_M)
        tiles_N = tl.cdiv(D, BLOCK_N)

        tiles_per_expert = tiles_M * tiles_N

        e = tile_id // tiles_per_expert
        local_tile_id = tile_id % tiles_per_expert

        tile_m = local_tile_id // tiles_N
        tile_n = local_tile_id % tiles_N

        dw_gate_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dw_up_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        f_offsets = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

        f_mask = f_offsets < F
        d_mask = d_offsets < D

        expert_start = tl.load(expert_offsets_ptr + e)
        expert_end = tl.load(expert_offsets_ptr + e + 1)
        N_e = expert_end - expert_start

        dw_positions = (
            e * F * D
            + f_offsets[:, None] * D
            + d_offsets[None, :]
        )

        for k_start in tl.range(0, N_e, BLOCK_K):
            local_k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = local_k_offsets < N_e
            global_k_offsets = expert_start + local_k_offsets

            x_positions = (
                global_k_offsets[:, None] * D
                + d_offsets[None, :]
            )

            x_tile = tl.load(
                x_ptr + x_positions,
                mask=k_mask[:, None] & d_mask[None, :],
                other = 0.0
            )

            dg_positions = (
                global_k_offsets[None, :] * F
                + f_offsets[:, None]
            )

            du_positions = dg_positions

            dg_tile = tl.load(
                dg_ptr + dg_positions,
                mask=f_mask[:, None] & k_mask[None, :],
                other = 0.0
            )

            du_tile = tl.load(
                du_ptr + du_positions,
                mask=f_mask[:, None] & k_mask[None, :],
                other=0.0
            )

            dw_gate_accum += tl.dot(dg_tile, x_tile)
            dw_up_accum += tl.dot(du_tile, x_tile)


        tl.store(
            dwgate_ptr + dw_positions,
            dw_gate_accum,
            mask=f_mask[:, None] & d_mask[None, :]
        )
        tl.store(
            dwup_ptr + dw_positions,
            dw_up_accum,
            mask=f_mask[:, None] & d_mask[None, :]
        )

        tile_id += num_workers



