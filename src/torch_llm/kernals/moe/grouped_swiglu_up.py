import triton
import torch as t
import triton.language as tl
from jaxtyping import Float, Int, Shaped
import math

from torch.nn import SiLU
from torch_llm.kernals.moe.scheduling import find_expert_for_tile_helper


def grouped_swiglu_up(
    x,
    gate_weight,
    up_weight,
    expert_offsets,
):
    """
    x:
        [A, d_model]

    gate_weight:
        [num_experts, d_ff, d_model]

    up_weight:
        [num_experts, d_ff, d_model]

    expert_offsets:
        [num_experts + 1]

    Returns eventually:
        hidden: [A, d_ff]
    """

    A, d_model = x.shape
    num_experts, d_ff, weight_d_model = gate_weight.shape

    assert weight_d_model == d_model
    assert up_weight.shape == gate_weight.shape
    assert expert_offsets.shape[0] == num_experts + 1

    hidden = t.empty((A, d_ff), device=x.device, dtype=x.dtype)

    BLOCK_M = 64 # number of expert-token rows handled by one program
    BLOCK_N = 64 # number of d_ff/output features handled by one program
    BLOCK_K = 32 # number of d_model/input features handled per chunk in program

    # Number of routed assignments belonging to each expert.
    #
    # expert_offsets:
    # [0, N_0, N_0 + N_1, ...]
    expert_counts = (
        expert_offsets[1:]
        - expert_offsets[:-1]
    )

    # Number of row tiles needed independently for each expert.
    tiles_m_per_expert = (
            expert_counts + BLOCK_M - 1
    ) // BLOCK_M

    # Every expert has the same d_ff width.
    tiles_n = triton.cdiv(d_ff, BLOCK_N)


    # Total 2D output tiles belonging to each expert.
    #
    # Each expert computes:
    #
    # [N_e, d_model] @ [d_model, d_ff]
    #
    # producing:
    #
    # [N_e, d_ff]
    tiles_per_expert = (
        tiles_m_per_expert * tiles_n
    )

    # Prefix sum over expert tile counts.
    # This will eventually let a global tile_id determine
    # which expert owns that tile.
    expert_tile_offsets = t.cat([
        t.zeros(
            1,
            device=x.device,
            dtype=tiles_per_expert.dtype,
        ),
        t.cumsum(tiles_per_expert, dim=0),
    ])

    # Total number of logical SwiGLU output tiles.
    total_tiles = int(
        expert_tile_offsets[-1].item()
    )

    if total_tiles == 0:
        return hidden

    #LATER TUNE PERHAPS
    desired_persistent_ctas = 36 # i have 36 SMs on my gpu, so 1 persistent cpa per sm. may not be optimal.
    num_workers = min(total_tiles, desired_persistent_ctas)
    grid = (num_workers,)

    SEARCH_STEPS = num_experts.bit_length() + 1 #math.ceil(math.log2(num_experts))


    grouped_swiglu_up_kernal[grid](
        x,
        gate_weight,
        up_weight,
        expert_offsets,
        expert_tile_offsets,
        hidden,

        d_model,
        d_ff,
        total_tiles,

        SEARCH_STEPS,
        num_experts,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return hidden


@triton.jit()
def grouped_swiglu_up_kernal(
        x_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        expert_offsets_ptr,
        expert_tile_offsets_ptr,
        hidden_ptr,

        d_model,
        d_ff,
        total_tiles,

        SEARCH_STEPS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    #M = assignment / token rows
    #N = d_ff output features
    #K = d_model reduction dimension


    pid = tl.program_id(0)
    num_workers = tl.num_programs(0)

    tile_id = pid

    while tile_id < total_tiles:
        # Find expert belonging to this GLOBAL TILE
        e = find_expert_for_tile_helper(
            expert_tile_offsets_ptr,
            tile_id,

            NUM_EXPERTS,
            SEARCH_STEPS,
        )

        # TOKEN offsets for this expert
        expert_offset = tl.load(expert_offsets_ptr + e)
        next_expert_offset = tl.load(expert_offsets_ptr + e + 1)
        #tokens_for_expert = next_expert_offset - expert_offset

        # TILE offset for this expert
        expert_tile_offset = tl.load(expert_tile_offsets_ptr + e)

        local_tile_id = tile_id - expert_tile_offset

        tile_N = tl.cdiv(d_ff, BLOCK_N)

        tile_m = local_tile_id // tile_N
        tile_n = local_tile_id % tile_N

        # Actual output rows / columns owned by this CTA
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
        n_mask = n_offsets < d_ff

        gate_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


        for k_start in tl.range(0, d_model, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < d_model

            x_positions = (
                    m_offsets[:, None] * d_model
                    + k_offsets[None, :]
            )

            x_tile = tl.load(
                x_ptr + x_positions,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            gate_positions = (
                    e * d_model * d_ff
                    + n_offsets[None, :] * d_model
                    + k_offsets[:, None]
            )

            gate_tile = tl.load(
                gate_weight_ptr + gate_positions,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            up_weight_positions = gate_positions

            up_tile = tl.load(
                up_weight_ptr + up_weight_positions,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            gate_accum += tl.dot(x_tile, gate_tile)
            up_accum += tl.dot(x_tile, up_tile)

        hidden = (gate_accum * tl.sigmoid(gate_accum)) * up_accum

        hidden_positions = (
                m_offsets[:, None] * d_ff
                + n_offsets[None, :]
        )

        tl.store(
            hidden_ptr + hidden_positions,
            hidden,
            mask=m_mask[:, None] & n_mask[None, :]
        )

        tile_id += num_workers







