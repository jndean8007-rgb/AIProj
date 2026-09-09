import math

import torch as t
from torch import nn
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int, Shaped
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl




class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            weight: Shaped[t.Tensor, "dmodel"] | None=None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        if weight is None:
            self.weight = nn.Parameter(t.ones(d_model))
        else:
            self.weight = nn.Parameter(weight)

    def forward(self, x: Shaped[t.Tensor, "T dmodel"]) -> Shaped[t.Tensor, "T dmodel"]:
        assert x.is_contiguous()
        assert x.ndim == 2
        assert self.weight.shape == (self.d_model,)
        return RMSNormFunction.apply(x, self.weight, self.eps)



class RMSNormFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        d_model = x.shape[-1]
        #need to calculate output AND rms from forwards
        output, inv_rms = triton_rmsnorm(x, weight, eps)
        ctx.save_for_backward(x, weight, inv_rms)
        return output

    @staticmethod
    @t.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()

        x, weight, inv_rms = ctx.saved_tensors

        dx, dw = triton_rmsnorm_backward_wrapper(x, weight, inv_rms, grad_output)

        return dx, dw, None



def triton_rmsnorm(x, weight, eps):
    # read shape of x
    num_rows = x.shape[0]
    # determine dmodel
    d_model = x.shape[1]
    # allocate output tensor and inv rms for backward
    output =t.empty_like(x)
    inv_rms = t.empty(num_rows, dtype=t.float32, device=x.device)
    # determine block size
    block_size = triton.next_power_of_2(d_model)
    # construct grid
    grid = (num_rows,)
    # launch kernal
    rmsnorm_kernal[grid](
        x,
        weight,
        output,
        inv_rms,
        d_model,
        eps,
        block_size
    )

    return output, inv_rms


@triton.jit
def rmsnorm_kernal(
        x_ptr,
        weight_ptr,
        output_ptr,
        inv_rms_ptr,
        d_model:tl.constexpr,
        eps: float,
        block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < d_model

    x = tl.load(x_ptr + pid * d_model + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sum_sqr = tl.sum(x * x, axis=0)

    mean_sqr = sum_sqr / d_model
    inv_rms = tl.rsqrt(mean_sqr + eps)

    x_norm = x * inv_rms

    x_elem = x_norm * weight

    tl.store(output_ptr + pid * d_model + offsets, x_elem, mask=mask)
    tl.store(inv_rms_ptr + pid , inv_rms)

#for non triton
# x**2:                 B*T*D elementwise ops
# reduction sum:        B*T*(D-1) additions
# scale + eps + sqrt:   O(B*T)
# normalize x:          B*T*D divisions
# weight scaling:       B*T*D multiplications

#benchmark stats of using triton vs regular
#RMSNORM BACKWARDS TIME HURRAY

def triton_rmsnorm_backward_wrapper(
        x: Shaped[t.Tensor, "T dmodel"],
        weight: Shaped[t.Tensor, "dmodel"],
        inv_rms: Shaped[t.Tensor, "T"],
        grad_output: Shaped[t.Tensor, "T dmodel"],
):

    num_rows = x.shape[0]
    d_model = x.shape[1]
    w_d_model = weight.shape[0]
    assert d_model == w_d_model
    assert x.shape == grad_output.shape
    assert num_rows == inv_rms.shape[0]

    dx = t.empty_like(x)
    BLOCK_M = 64 #each program owns 1 feature and loops through dmodel in chunks of 64
    grid = (num_rows,)

    triton_rmsnorm_backward_dx[grid](
        grad_output,
        x,
        weight,
        inv_rms,
        dx,

        num_rows,
        d_model,

        BLOCK_M
    )

    dw = t.empty((d_model,), dtype=x.dtype, device=x.device)
    BLOCK_M = 64 #each program owns dmodel feature 1 and loops through tokens in blockm chunks
    grid = (d_model,)

    triton_rmsnorm_backward_dw[grid](
        grad_output,
        x,
        inv_rms,
        dw,

        num_rows,
        d_model,

        BLOCK_M
    )

    return dx, dw




@triton.jit()
def triton_rmsnorm_backward_dx(
        grad_output_ptr, #T, D
        x_ptr,
        weight_ptr, #D
        inv_rms_ptr, #T
        dx_ptr, #T, D

        context_len, #T
        d_model, #D

        BLOCK_M: tl.constexpr #D
): #reduction over dmodel features in dX blockm chunks NEED TO REWRITE: DX NEED ENTIRE DOT FOR REDUCTION
    token_num = tl.program_id(0)

    inv_rms = tl.load(inv_rms_ptr + token_num)
    #calculate dot accum
    dot_accum = tl.zeros((), dtype=tl.float32)

    for m_start in tl.range(0, d_model, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < d_model

        weight_positions = m_offsets

        weight_tile = tl.load(
            weight_ptr + weight_positions,
            mask=m_mask,
            other=0.0,
        )

        grad_output_positions = token_num * d_model + m_offsets
        x_positions = grad_output_positions

        grad_output_tile = tl.load(
            grad_output_ptr + grad_output_positions,
            mask=m_mask,
            other=0.0,
        )

        x_tile = tl.load(
            x_ptr + x_positions,
            mask=m_mask,
            other=0.0,
        )

        h_tile = grad_output_tile * weight_tile


        dot_accum += tl.sum(h_tile * x_tile, axis=0)

    dot = dot_accum


    for m_start in tl.range(0, d_model, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < d_model

        weight_positions = m_offsets

        weight_tile = tl.load(
            weight_ptr + weight_positions,
            mask=m_mask,
            other=0.0,
        )

        grad_output_positions = token_num * d_model + m_offsets
        x_positions = grad_output_positions

        grad_output_tile = tl.load(
            grad_output_ptr + grad_output_positions,
            mask=m_mask,
            other=0.0,
        )

        x_tile =tl.load(
            x_ptr + x_positions,
            mask=m_mask,
            other=0.0,
        )

        h_tile = grad_output_tile * weight_tile

        dx_chunk = h_tile * inv_rms - x_tile * (inv_rms * inv_rms * inv_rms / d_model) * dot

        dx_positions = token_num * d_model + m_offsets
        tl.store(dx_ptr + dx_positions, dx_chunk, mask=m_mask)



@triton.jit()
def triton_rmsnorm_backward_dw(
        grad_output_ptr, #T D
        x_ptr, #T D
        inv_rms_ptr, #T
        dw_ptr, #D

        context_len, #T
        d_model, #D

        BLOCK_M: tl.constexpr #T
): #reduction over tokens in T blockm chunks
    feature_id = tl.program_id(0)

    dw_accum = tl.zeros((), dtype=tl.float32)

    for m_start in tl.range(0, context_len, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < context_len

        positions = m_offsets * d_model + feature_id

        grad_output_tile = tl.load(
            grad_output_ptr + positions,
            mask=m_mask,
            other=0.0,
        )

        x_tile = tl.load(
            x_ptr + positions,
            mask=m_mask,
            other=0.0,
        )

        inv_rms_tile = tl.load(
            inv_rms_ptr + m_offsets,
            mask=m_mask,
            other=0.0,
        )

        dw_accum += tl.sum(
            grad_output_tile * x_tile * inv_rms_tile,
            axis=0
        )

    tl.store(dw_ptr + feature_id, dw_accum)
