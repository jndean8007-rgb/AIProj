import torch as t
import torch.nn.functional as F

from torch_llm.model.moe.moe_autograd import MoeAutograd  # adapt import/name


def reference_grouped_experts(
    x,
    gate_weight,
    up_weight,
    down_weight,
    expert_offsets,
):
    """
    x:             (A, D), already expert-major
    gate_weight:   (E, F, D)
    up_weight:     (E, F, D)
    down_weight:   (E, D, F)
    """

    outputs = []

    num_experts = gate_weight.shape[0]

    for e in range(num_experts):
        start = int(expert_offsets[e])
        end = int(expert_offsets[e + 1])

        x_e = x[start:end]

        gate = F.linear(x_e, gate_weight[e])
        up = F.linear(x_e, up_weight[e])

        hidden = F.silu(gate) * up

        y_e = F.linear(hidden, down_weight[e])

        outputs.append(y_e)

    return t.cat(outputs, dim=0)


def test_grouped_expert_backward():
    t.manual_seed(0)

    device = "cuda"
    dtype = t.float16

    # Small but deliberately ragged:
    # - zero-token expert
    # - tiny expert
    # - multiple experts
    expert_counts = t.tensor(
        [5, 0, 9, 3],
        device=device,
        dtype=t.int32,
    )

    expert_offsets = t.cat([
        t.zeros(1, device=device, dtype=t.int32),
        expert_counts.cumsum(0),
    ])

    A = int(expert_offsets[-1].item())
    E = expert_counts.numel()

    D = 64
    F_DIM = 128

    # ----------------------------
    # Custom Triton path
    # ----------------------------

    x = t.randn(
        A, D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    gate_weight = t.randn(
        E, F_DIM, D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    ) * 0.05

    up_weight = t.randn(
        E, F_DIM, D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    ) * 0.05

    down_weight = t.randn(
        E, D, F_DIM,
        device=device,
        dtype=dtype,
        requires_grad=True,
    ) * 0.05

    # Make them leaf tensors after scaling.
    gate_weight = gate_weight.detach().requires_grad_(True)
    up_weight = up_weight.detach().requires_grad_(True)
    down_weight = down_weight.detach().requires_grad_(True)

    y = MoeAutograd.apply(
        x,
        gate_weight,
        up_weight,
        down_weight,
        expert_offsets,
    )

    grad_y = t.randn_like(y)

    y.backward(grad_y)

    custom_dx = x.grad.detach().clone()
    custom_dwg = gate_weight.grad.detach().clone()
    custom_dwu = up_weight.grad.detach().clone()
    custom_dwd = down_weight.grad.detach().clone()

    # ----------------------------
    # Pure PyTorch reference
    # ----------------------------

    x_ref = x.detach().clone().requires_grad_(True)
    wg_ref = gate_weight.detach().clone().requires_grad_(True)
    wu_ref = up_weight.detach().clone().requires_grad_(True)
    wd_ref = down_weight.detach().clone().requires_grad_(True)

    y_ref = reference_grouped_experts(
        x_ref,
        wg_ref,
        wu_ref,
        wd_ref,
        expert_offsets,
    )

    y_ref.backward(grad_y)

    # ----------------------------
    # Compare
    # ----------------------------

    t.testing.assert_close(
        custom_dx,
        x_ref.grad,
        rtol=3e-2,
        atol=3e-2,
    )

    t.testing.assert_close(
        custom_dwg,
        wg_ref.grad,
        rtol=3e-2,
        atol=3e-2,
    )

    t.testing.assert_close(
        custom_dwu,
        wu_ref.grad,
        rtol=3e-2,
        atol=3e-2,
    )

    t.testing.assert_close(
        custom_dwd,
        wd_ref.grad,
        rtol=3e-2,
        atol=3e-2,
    )
