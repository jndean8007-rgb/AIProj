import torch
import torch as t
import triton
import triton.language as tl

from torch_llm.kernals.moe.grouped_down_bwd import grouped_down_input_grad, grouped_down_weight_grad
from torch_llm.kernals.moe.grouped_swiglu_bwd import grouped_swiglu_backward
from torch_llm.kernals.moe.grouped_swiglu_up import grouped_swiglu_up
from torch_llm.kernals.moe.grouped_down import grouped_down

class MoeAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            expert_inputs,
            gate_weight,
            up_weight,
            down_weight,
            expert_offsets
    ):
        hidden = grouped_swiglu_up(
            expert_inputs,
            gate_weight,
            up_weight,
            expert_offsets
        )

        ctx.save_for_backward(expert_inputs, hidden, gate_weight, up_weight, down_weight, expert_offsets)

        output = grouped_down(
            hidden,
            down_weight,
            expert_offsets
        )

        return output

    # returns expert outputs

    @staticmethod
    @t.autograd.function.once_differentiable
    def backward(
            ctx,
            grad_expert_outputs
    ):
        grad_expert_outputs = grad_expert_outputs.contiguous()

        expert_inputs, hidden, gate_weight, up_weight, down_weight, expert_offsets = ctx.saved_tensors

        dh = grouped_down_input_grad(
            grad_expert_outputs,
            down_weight,
            expert_offsets
        )

        dw_down = grouped_down_weight_grad(
            grad_expert_outputs,
            hidden,
            expert_offsets
        )

        dx, dw_gate, dw_up = grouped_swiglu_backward(
            dh,
            gate_weight,
            up_weight,
            expert_inputs,
            expert_offsets
        )

        return dx, dw_gate, dw_up, dw_down, None



