import torch as t
import triton
import triton.language as tl

@triton.jit()
def find_expert_for_tile_helper(
        expert_tile_offsets_ptr,

        tile_id,

        NUM_EXPERTS: tl.constexpr,
        SEARCH_STEPS: tl.constexpr
):
    low = 0
    high = NUM_EXPERTS
    for _ in tl.static_range(0, SEARCH_STEPS):

        mid = (low + high) // 2
        go_right = tile_id >= tl.load(expert_tile_offsets_ptr + mid+1)

        #swithc to tl.where
        low = tl.where(go_right, mid + 1, low)
        high = tl.where(go_right, high, mid)

    return low