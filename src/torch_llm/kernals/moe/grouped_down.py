import triton
import torch as t
import triton.language as tl
from jaxtyping import Float, Int, Shaped
import math
from torch_llm.kernals.moe.scheduling import find_expert_for_tile_helper

def grouped_down(
        hidden,
        down_weight,
        expert_offsets
) -> Shaped[t.Tensor, 'A d_model']:

    A, d_ff = hidden.shape
    num_experts, d_model, weight_d_ff = down_weight.shape

    assert d_ff == weight_d_ff
    assert expert_offsets.shape[0] == num_experts + 1


    #dimension substitution compared to swiglu up: K = d_ff and N = d_model instead of other way around
    BLOCK_M = 64 # number of expert-token rows handled by one program
    BLOCK_N = 32 # number of d_model/output features handled by one program
    BLOCK_K = 64 # number of d_ff/input features handled per chunk in program

    expert_counts = (
        expert_offsets[1:]
        - expert_offsets[:-1]
    )

    tiles_m_per_expert = (
            expert_counts + BLOCK_M - 1
    ) // BLOCK_M

    tiles_n = triton.cdiv(d_model, BLOCK_N)


    tiles_per_expert = (
        tiles_m_per_expert * tiles_n
    )

    expert_tile_offsets = t.cat([
        t.zeros(
            1,
            device=hidden.device,
            dtype=tiles_per_expert.dtype
        ),
        t.cumsum(tiles_per_expert, dim=0)
    ])
    # Prefix sum of tile counts per expert.
    # Lets a global tile_id determine which expert owns that output tile.
    #
    # BLOCK_K does not appear here because K is not part of the global output-tile grid.
    # Each Triton program owns one [BLOCK_M, BLOCK_N] output tile and loops over
    # d_ff in BLOCK_K chunks internally

    total_tiles = int(
        expert_tile_offsets[-1].item()
    )

    if total_tiles == 0:
        return hidden

    desired_persistent_ctas = 36
    num_workers = min(desired_persistent_ctas, total_tiles)
    grid = (num_workers,)

    SEARCH_STEPS = num_experts.bit_length() + 1 #math.ceil(math.log2(num_experts))

    output = t.empty((A, d_model), device=hidden.device, dtype=hidden.dtype)

    grouped_down_kernal[grid](
        hidden,
        down_weight,
        expert_offsets,
        expert_tile_offsets,
        output,

        d_ff,
        d_model,
        total_tiles,

        SEARCH_STEPS,
        num_experts,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return output

@triton.jit()
def grouped_down_kernal(
        hidden_ptr,
        down_weight_ptr,
        expert_offsets_ptr,
        expert_tile_offsets_ptr,
        output_ptr,

        d_ff,
        d_model,
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

        tile_N = tl.cdiv(d_model, BLOCK_N)

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
        n_mask = n_offsets < d_model

        out_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, d_ff, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < d_ff

            hidden_positions = (
                    m_offsets[:, None] * d_ff +
                    k_offsets[None, :]
            )

            hidden_tile = tl.load(
                hidden_ptr + hidden_positions,
                mask = m_mask[:, None] & k_mask[None, :],
                other = 0.0
            ) #block m, block k, so want to load weights as block k, block n for convenience

            down_weight_positions = ( #e, d_model, d_ff as pytorch convention for rowwise correspondence
                e * d_model * d_ff
                + n_offsets[None, :] * d_ff
                + k_offsets[:, None]
            ) #REMEMBER N IS DMODEL, AND CONVENTION WE ARE USING IS (E, DMODEL, DFF)

            down_weight_tile = tl.load(
                down_weight_ptr + down_weight_positions,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            # down_weight_tile is [BLOCK_K, BLOCK_N]:
            # rows index d_ff (K), columns index d_model (N)
            # k_mask[:, None] masks rows
            # n_mask[None, :] masks columns

            out_accum += tl.dot(hidden_tile, down_weight_tile)


        out_positions = (
            m_offsets[:, None] * d_model
            + n_offsets[None, :]
        )

        tl.store(
            output_ptr + out_positions,
            out_accum,
            mask = m_mask[:, None] & n_mask[None, :],
        )

        tile_id += num_workers