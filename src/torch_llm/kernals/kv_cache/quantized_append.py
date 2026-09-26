import torch as t
import math
import triton
import triton.language as tl


'''
token × KV head

load Dh values
→ max(abs(...))
→ calculate scale
→ quantize Dh values
→ store quantized vector
→ store one scale

k[physical_block, offset, head, :]
k_scales[physical_block, offset, head]

v[physical_block, offset, head, :]
v_scales[physical_block, offset, head]
'''

'''
contract:
loads in head dimension values for a particular block, block offset, head.
gets max value. calculates scale: scale = max abs d / t.finfo(cache_dtype).ax
p = quantize(x / s)

x^ = p * s

write both x^ and scale
'''

@triton.jit
def quantize_append(
        k_cache_ptr,
        v_scale_ptr,
        v_scale_ptr,
        x_ptr,

        cache_slots_ptr,
        context_lengths_ptr,
        block_table_ptr,

        cum_splits_ptr,
        split_batch_ids_ptr,

        #num_heads?/????

        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,

):

