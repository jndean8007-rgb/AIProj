import triton
import torch as t
import triton.language as tl
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int, Shaped
import math



class FlashAttentionFunction(t.autograd.Function): #operates on T h d where T is cumulative sequence lengths over batch ids, shape (B+1,)
    @staticmethod
    def forward(ctx, q, k, v, cum_seq: Shaped[t.Tensor, "b"], batch_max_seq_len): #, causal, kv_cache=None DO THE KV CACHE
        # Triton forward should return both tensors.

        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        output, lse = flash_attention_forward(q, k, v, cum_seq, batch_max_seq_len, scale)

        ctx.save_for_backward(q, k, v, output, lse, cum_seq)
        #ctx.causal = causal NOT YET, WE WILL IMPLEMENT IN THE FUTURE FOR MULTIDIRECTIONAL ATTENTION
        ctx.batch_max_seq_len = batch_max_seq_len
        ctx.scale = scale
        return output

    @staticmethod
    @t.autograd.function.once_differentiable
    def backward(ctx, grad_output): #returns dq, dk, dv, None, None for use in pytorch backwards calculation IE dL/dOutput -> dL/dQuery etc.
        q, k, v, output, lse, cum_seq = ctx.saved_tensors
        #causal = ctx.causal NOT YET, WE WILL IMPLEMENT IN THE FUTURE FOR MULTIDIRECTIONAL ATTENTION
        grad_output = grad_output.contiguous()
        batch_max_seq_len = ctx.batch_max_seq_len
        scale = ctx.scale

        dq, dk, dv = flash_attention_backward(
            q,
            k,
            v,
            output,
            lse,
            cum_seq,
            grad_output,
            batch_max_seq_len,
            scale
        )

        return dq, dk, dv, None, None




def flash_attention_forward(q: Shaped[t.Tensor, "T hq d"],
                            k: Shaped[t.Tensor, "T hkv d"],
                            v: Shaped[t.Tensor, "T hkv d"],
                            cum_seq: Shaped[t.Tensor, "b"],
                            batch_max_seq_len,
                            scale)\
        -> tuple[Shaped[t.Tensor, "T hq d"], Shaped[t.Tensor, "T hq"]]:

    q_shape, k_shape, v_shape = q.shape, k.shape, v.shape
    assert k.shape == v.shape
    assert q.shape[0] == k.shape[0]
    assert q.shape[2] == k.shape[2]

    T, queryH, D = q.shape
    kvH = k.shape[1]
    B = cum_seq.shape[0] - 1

    assert queryH % kvH == 0

    BLOCK_M = 64
    BLOCK_N = 64
    HEAD_DIM = triton.next_power_of_2(D)
    max_query_blocks = triton.cdiv(batch_max_seq_len, BLOCK_M)

    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    grid = (max_query_blocks, B * queryH)

    output = t.empty_like(q)
    lse = t.empty((T, queryH), device=q.device, dtype = t.float32)

    flash_attention_forward_kernal[grid](
        q,
        k,
        v,
        output,
        lse,

        cum_seq,
        queryH,
        kvH,
        D,
        scale,

        BLOCK_M,
        BLOCK_N,
        HEAD_DIM
    )
    return output, lse



@triton.jit
def flash_attention_forward_kernal(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        lse_ptr,
        cum_seq_ptr,
        
        num_query_heads,
        num_kv_heads,
        head_dim,
        scale,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):

    query_block_id = tl.program_id(axis=0)
    batch_head_id  = tl.program_id(axis=1)

    batch_id = batch_head_id // num_query_heads
    query_head_id = batch_head_id % num_query_heads

    query_heads_per_kv_head = num_query_heads // num_kv_heads
    kv_head_id = query_head_id // query_heads_per_kv_head
    
    request_start = tl.load(cum_seq_ptr + batch_id) # retrieves start token in overall sequence
    request_end = tl.load(cum_seq_ptr + batch_id + 1) # end token
    request_len = request_end - request_start

    local_query_offsets = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_query_offsets = request_start + local_query_offsets

    query_mask = local_query_offsets < request_len
    query_row_base = (packed_query_offsets * num_query_heads + query_head_id) * head_dim

    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < head_dim
    query_load_mask = query_mask[:, None] & head_mask[None, :]

    feature_positions = query_row_base[:, None] + head_offsets[None, :]
    q_tile = tl.load(q_ptr + feature_positions, mask = query_load_mask, other=0.0)

    # both are dim block_m. because each query row needs a largest score seen and a sum of exponentials seen
    running_max = tl.full((BLOCK_M,), -float("inf"), dtype = tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype = tl.float32)

    #shape of is blockm, head_dim because a full output vector is outputted per output row
    output_accum = tl.zeros((BLOCK_M, HEAD_DIM), dtype = tl.float32)

    lse_positions = (packed_query_offsets * num_query_heads + query_head_id)

    for kv_start in tl.range(0, request_len, BLOCK_N):
        local_kv_offsets = kv_start + tl.arange(0, BLOCK_N)
        packed_kv_offsets = request_start + local_kv_offsets

        kv_base = ((packed_kv_offsets * num_kv_heads + kv_head_id) * head_dim)
        kv_mask = local_kv_offsets < request_len

        kv_positions = kv_base[:, None] + head_offsets[None, :]
        kv_load_mask = kv_mask[:, None] & head_mask[None, :]

        k_tile = tl.load(k_ptr + kv_positions, mask = kv_load_mask, other=0.0)

        # shape is block_n, head_dim
        v_tile = tl.load(v_ptr + kv_positions, mask = kv_load_mask, other=0.0)

        scores = tl.dot(q_tile, tl.trans(k_tile))

        scores *= scale

        causal_mask = local_kv_offsets[None, :] <= local_query_offsets[:, None]
        valid_kv_mask = kv_mask[None, :]

        score_mask = causal_mask & valid_kv_mask & query_mask[:, None]
        #(M, N)
        scores = tl.where(
            score_mask,
            scores,
            -float("inf")
        )
        #(M,)
        block_max = tl.max(scores, axis = -1)
        #(M,)
        new_running_max = tl.maximum(running_max, block_max)
        correction_factor = tl.exp(running_max - new_running_max)

        current_exp = tl.exp(scores - new_running_max[:, None])
        current_sum = tl.sum(current_exp, axis=-1)

        new_running_sum = correction_factor * running_sum + current_sum

        #accumulating num and denom seperately
        current_output_contribution = tl.dot(current_exp.to(v_tile.dtype), v_tile)
        new_output_accumulator = output_accum * correction_factor[:, None] + current_output_contribution

        output_accum = new_output_accumulator
        running_sum = new_running_sum
        running_max = new_running_max

    final_output = output_accum / running_sum[:, None]
    lse_final = running_max + tl.log(running_sum)

    tl.store(output_ptr + feature_positions, final_output, mask = query_load_mask)
    tl.store(lse_ptr + lse_positions, lse_final, mask=query_mask)


def flash_attention_backward(q,
                             k,
                             v,
                             output,
                             lse,
                             cum_seq,
                             grad_output,
                             batch_max_seq_len,
                             scale): # ------------------ time to work on this to make work for ragged prompts. not very hard, so dont worry.
    delta = flash_attention_backward_preprocess_wrapper(output, grad_output, cum_seq, batch_max_seq_len)
    T, qh, d = q.shape
    kvh = k.shape[-2]
    BLOCK_M = 64
    BLOCK_N = 64
    HEAD_DIM = triton.next_power_of_2(d)
    b = cum_seq.shape[0] - 1
    assert qh % kvh == 0
    assert k.shape == v.shape
    assert q.shape[0] == k.shape[0]  # same T
    assert q.shape[2] == k.shape[2]  # same head dimension
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    griddq = (triton.cdiv(batch_max_seq_len, BLOCK_M), b * qh)
    dq = t.empty_like(q)
    flash_attention_backward_dq[griddq](
        q,
        k,
        v,
        grad_output,
        lse,
        delta,
        dq,
        cum_seq,

        qh,
        kvh,
        d,
        scale,

        BLOCK_M,
        BLOCK_N,
        HEAD_DIM
    )

    griddkdv = (triton.cdiv(batch_max_seq_len, BLOCK_N), b * kvh)
    dk = t.empty_like(k)
    dv = t.empty_like(v)
    flash_attention_backward_dkdv[griddkdv](
        q,
        k,
        v,
        grad_output,
        lse,
        delta,
        dk,
        dv,
        cum_seq,

        qh,
        kvh,
        d,
        scale,

        BLOCK_M,
        BLOCK_N,
        HEAD_DIM
    )

    return dq, dk, dv

def flash_attention_backward_preprocess_wrapper(output, grad_output, cum_seq, batch_max_seq_len) -> Shaped[t.Tensor, "t h"]: #input shapes FOR ALL KERNALS are THD
    T, hq, d = output.shape
    b = cum_seq.shape[0] - 1

    HEAD_DIM = triton.next_power_of_2(d)
    BLOCK_M = 64
    grid = (triton.cdiv(batch_max_seq_len, BLOCK_M), b * hq)
    delta = t.empty((T, hq), device=output.device, dtype=t.float32)

    flash_attention_backward_preprocess[grid](
        output,
        grad_output,
        delta,
        cum_seq,

        hq,
        d,

        BLOCK_M,
        HEAD_DIM
    )

    return delta


@triton.jit
def flash_attention_backward_preprocess(
        output_ptr,
        grad_output_ptr,
        delta_ptr,
        cum_seq_ptr,

        query_num_heads,
        head_dim,

        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
): #outputs delta -> sum of OdO element wise NOT KRONECKER DELTA ----- everything is named query because im lazy and didn't change to output.
    query_block_id = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1)

    batch_id = batch_head_id // query_num_heads
    query_head_id = batch_head_id % query_num_heads

    request_start = tl.load(cum_seq_ptr + batch_id)
    request_end = tl.load(cum_seq_ptr + batch_id + 1)  # end token
    request_len = request_end - request_start

    local_query_offsets = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_query_offsets = request_start + local_query_offsets
    query_mask = local_query_offsets < request_len
    query_row_base = (packed_query_offsets * query_num_heads + query_head_id) * head_dim

    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < head_dim
    load_mask = query_mask[:, None] & head_mask[None, :]

    positions = query_row_base[:, None] + head_offsets[None, :]

    O_tile = tl.load(output_ptr + positions, mask = load_mask, other=0.0)
    dO_tile = tl.load(grad_output_ptr + positions, mask = load_mask, other=0.0)

    element_wise = O_tile * dO_tile
    delta = tl.sum(element_wise, axis = -1) #locally (BLOCK_M, ), globally (T, hq)
    delta_pos = packed_query_offsets * query_num_heads + query_head_id

    tl.store(delta_ptr + delta_pos, delta, mask=query_mask)


@triton.jit
def flash_attention_backward_dq(
        q_ptr,
        k_ptr,
        v_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        dq_ptr,
        cum_seq_ptr,

        num_query_heads,
        num_kv_heads,
        head_dim,
        scale,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    query_block_id = tl.program_id(axis=0)
    batch_query_head_id = tl.program_id(axis=1)

    batch_id = batch_query_head_id // num_query_heads
    query_head_id = batch_query_head_id % num_query_heads

    query_heads_per_kv_head = num_query_heads // num_kv_heads
    kv_head_id = query_head_id // query_heads_per_kv_head

    request_start = tl.load(cum_seq_ptr + batch_id)
    request_end = tl.load(cum_seq_ptr + batch_id + 1)
    request_len = request_end - request_start

    local_query_offsets = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    query_mask = local_query_offsets < request_len
    head_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < head_dim

    packed_query_offsets = request_start + local_query_offsets
    query_row_base = (packed_query_offsets * num_query_heads + query_head_id) * head_dim
    load_mask = query_mask[:, None] & head_mask[None, :]
    query_items = query_row_base[:, None] + head_offsets[None, :]

    q_tile = tl.load(q_ptr + query_items, mask = load_mask, other=0.0)
    dO_tile = tl.load(grad_output_ptr + query_items, mask = load_mask, other=0.0)

    lse_positions = (packed_query_offsets * num_query_heads + query_head_id)
    lse = tl.load(lse_ptr + lse_positions, mask = query_mask, other=0.0)
    delta = tl.load(delta_ptr + lse_positions, mask = query_mask, other=0.0)

    dQ_accum = tl.zeros((BLOCK_M, HEAD_DIM), dtype = tl.float32)

    for kv_start in tl.range(0, request_len, BLOCK_N):
        local_kv_offsets = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = local_kv_offsets < request_len
        kv_load_mask = kv_mask[:, None] & head_mask[None, :]

        packed_kv_offsets = request_start + local_kv_offsets
        kv_row_base = (packed_kv_offsets * num_kv_heads + kv_head_id) * head_dim
        kv_positions = kv_row_base[:, None] + head_offsets[None, :]

        k_tile = tl.load(k_ptr + kv_positions, mask = kv_load_mask, other=0.0)
        v_tile = tl.load(v_ptr + kv_positions, mask = kv_load_mask, other=0.0)

        s_tile = scale * tl.dot(q_tile, tl.trans(k_tile)) #ktile: (BLOCK_N, headdim) -> stile: BLOCK_M, BLOCK_N
        causal_mask = local_kv_offsets[None, :] <= local_query_offsets[:, None]
        valid_kv_mask = kv_mask[None, :]

        score_mask = causal_mask & valid_kv_mask & query_mask[:, None]

        s_tile = tl.where(score_mask, s_tile, -float('inf'))

        p_tile = tl.exp(s_tile - tl.unsqueeze(lse, 1))

        dP_tile = tl.dot(dO_tile, tl.trans(v_tile))

        dS_tile = p_tile * (dP_tile - tl.unsqueeze(delta, 1))

        dQ_tile = scale * tl.dot(dS_tile, k_tile.to(dS_tile.dtype))

        dQ_accum += dQ_tile

    tl.store(dq_ptr + query_items, dQ_accum, mask=load_mask)

@triton.jit
def flash_attention_backward_dkdv( # grid = (t.cdiv(s / block_n), b * kv_head)
        q_ptr,
        k_ptr,
        v_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        dk_ptr,
        dv_ptr,
        cum_seq_ptr,

        num_query_heads,
        num_kv_heads,
        head_dim,
        scale,

        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr
):
    kv_block_id = tl.program_id(axis=0)
    batch_kv_head_id = tl.program_id(axis=1)

    batch_id = batch_kv_head_id // num_kv_heads
    kv_head_id = batch_kv_head_id % num_kv_heads

    request_start = tl.load(cum_seq_ptr + batch_id)  # retrieves start token in overall sequence
    request_end = tl.load(cum_seq_ptr + batch_id + 1)  # end token
    request_len = request_end - request_start

    local_kv_offsets = kv_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    head_offsets = tl.arange(0, HEAD_DIM)

    kv_mask = local_kv_offsets < request_len
    head_mask = head_offsets < head_dim
    packed_kv_offsets = request_start + local_kv_offsets

    kv_row_base = (packed_kv_offsets * num_kv_heads + kv_head_id) * head_dim
    kv_positions = kv_row_base[:, None] + head_offsets[None, :]
    kv_load_mask = kv_mask[:, None] & head_mask[None, :]

    k_tile = tl.load(k_ptr + kv_positions, mask = kv_load_mask, other=0.0)
    v_tile = tl.load(v_ptr + kv_positions, mask = kv_load_mask, other=0.0)

    dK_accum = tl.zeros((BLOCK_N, HEAD_DIM), dtype = tl.float32)
    dV_accum = tl.zeros((BLOCK_N, HEAD_DIM), dtype = tl.float32)

    query_heads_per_kv_head = num_query_heads // num_kv_heads
    first_query_head = query_heads_per_kv_head * kv_head_id

    for query_head_id in tl.range(first_query_head, first_query_head + query_heads_per_kv_head):

        for q_start in tl.range(0, request_len, BLOCK_M):
            local_query_offsets = q_start + tl.arange(0, BLOCK_M)
            query_mask = local_query_offsets < request_len
            packed_q_offsets = request_start + local_query_offsets

            qdO_row_base = (packed_q_offsets * num_query_heads + query_head_id) * head_dim
            qdO_positions = qdO_row_base[:, None] + head_offsets[None, :]

            load_mask = query_mask[:, None] & head_mask[None, :]

            q_tile = tl.load(q_ptr + qdO_positions, mask = load_mask, other=0.0)
            dO_tile = tl.load(grad_output_ptr + qdO_positions, mask = load_mask, other=0.0)

            dlse_positions = (packed_q_offsets * num_query_heads + query_head_id)
            delta_tile = tl.load(delta_ptr + dlse_positions, mask = query_mask, other=0.0)
            lse_tile = tl.load(lse_ptr + dlse_positions, mask = query_mask, other=0.0)

            s_tile = scale * tl.dot(q_tile, tl.trans(k_tile))
            causal_mask = local_kv_offsets[None, :] <= local_query_offsets[:, None]
            valid_kv_mask = kv_mask[None, :]
            score_mask = causal_mask & valid_kv_mask & query_mask[:, None]

            s_tile = tl.where(score_mask, s_tile, -float('inf')) #blockm blockn

            p_tile = tl.exp(s_tile - tl.unsqueeze(lse_tile, 1)) #blockm 1 :lse -> blockm blockn

            dV_contribution = tl.dot(tl.trans(p_tile), dO_tile.to(p_tile.dtype)) #blockn blockm @ blockm d -> blockn d
            dV_accum += dV_contribution
            dP_tile = tl.dot(dO_tile, tl.trans(v_tile))
            dS_tile = p_tile * (dP_tile - tl.unsqueeze(delta_tile, 1))

            dK_contribution = scale * tl.dot(tl.trans(dS_tile), q_tile.to(dS_tile.dtype))
            dK_accum += dK_contribution

    tl.store(dk_ptr + kv_positions, dK_accum, mask=kv_load_mask)
    tl.store(dv_ptr + kv_positions, dV_accum, mask=kv_load_mask)


