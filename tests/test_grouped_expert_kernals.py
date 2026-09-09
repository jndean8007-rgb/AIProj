import torch as t
import torch.nn.functional as F

# Adjust these imports to your project
from torch_llm.kernals.moe.grouped_swiglu_up import grouped_swiglu_up
from torch_llm.kernals.moe.grouped_down import grouped_down

# py -m pytest tests/test_grouped_expert_kernals.py -v

DEVICE = "cuda"
DTYPE = t.float16


def make_test_data():
    """
    Deliberately ugly expert loads.

    Includes:
        - zero-token experts
        - one-token expert
        - partial BLOCK_M tiles
        - multiple BLOCK_M tiles
    """
    t.manual_seed(0)
    t.cuda.manual_seed_all(0)

    expert_counts = t.tensor(
        [0, 1, 17, 65, 129, 3, 0, 78],
        device=DEVICE,
        dtype=t.int32,
    )

    num_experts = expert_counts.numel()

    expert_offsets = t.cat([
        t.zeros(1, device=DEVICE, dtype=t.int32),
        t.cumsum(expert_counts, dim=0),
    ])

    A = int(expert_counts.sum().item())

    d_model = 128
    d_ff = 256

    # Inputs are ALREADY in expert-major order.
    expert_inputs = t.randn(
        A,
        d_model,
        device=DEVICE,
        dtype=DTYPE,
    )

    # Same layouts as your ExpertWeights class.
    gate_weight = t.randn(
        num_experts,
        d_ff,
        d_model,
        device=DEVICE,
        dtype=DTYPE,
    ) / d_model**0.5

    up_weight = t.randn(
        num_experts,
        d_ff,
        d_model,
        device=DEVICE,
        dtype=DTYPE,
    ) / d_model**0.5

    down_weight = t.randn(
        num_experts,
        d_model,
        d_ff,
        device=DEVICE,
        dtype=DTYPE,
    ) / d_ff**0.5

    return (
        expert_inputs,
        gate_weight,
        up_weight,
        down_weight,
        expert_offsets,
        num_experts,
        d_model,
        d_ff,
    )


def reference_swiglu(
    expert_inputs,
    gate_weight,
    up_weight,
    expert_offsets,
):
    """
    Slow PyTorch correctness oracle.

    Returns:
        hidden: (A, d_ff)
    """
    A = expert_inputs.shape[0]
    d_ff = gate_weight.shape[1]
    num_experts = gate_weight.shape[0]

    hidden = t.empty(
        A,
        d_ff,
        device=expert_inputs.device,
        dtype=expert_inputs.dtype,
    )

    for e in range(num_experts):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())

        if start == end:
            continue

        x_e = expert_inputs[start:end]

        gate = F.linear(
            x_e,
            gate_weight[e],
        )

        up = F.linear(
            x_e,
            up_weight[e],
        )

        hidden[start:end] = F.silu(gate) * up

    return hidden


def reference_down(
    hidden,
    down_weight,
    expert_offsets,
):
    """
    Slow PyTorch correctness oracle.

    Returns:
        output: (A, d_model)
    """
    A = hidden.shape[0]
    d_model = down_weight.shape[1]
    num_experts = down_weight.shape[0]

    output = t.empty(
        A,
        d_model,
        device=hidden.device,
        dtype=hidden.dtype,
    )

    for e in range(num_experts):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())

        if start == end:
            continue

        output[start:end] = F.linear(
            hidden[start:end],
            down_weight[e],
        )

    return output


def test_grouped_swiglu_up():
    (
        expert_inputs,
        gate_weight,
        up_weight,
        _,
        expert_offsets,
        *_,
    ) = make_test_data()

    expected = reference_swiglu(
        expert_inputs,
        gate_weight,
        up_weight,
        expert_offsets,
    )

    actual = grouped_swiglu_up(
        expert_inputs,
        gate_weight,
        up_weight,
        expert_offsets,
    )

    t.cuda.synchronize()

    print(
        "SwiGLU max abs error:",
        (actual - expected).abs().max().item(),
    )

    assert actual.shape == expected.shape
    assert t.isfinite(actual).all()

    t.testing.assert_close(
        actual,
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_grouped_down():
    (
        expert_inputs,
        gate_weight,
        up_weight,
        down_weight,
        expert_offsets,
        *_,
    ) = make_test_data()

    # Use PyTorch reference hidden so this test isolates
    # the DOWN kernel.
    hidden = reference_swiglu(
        expert_inputs,
        gate_weight,
        up_weight,
        expert_offsets,
    )

    expected = reference_down(
        hidden,
        down_weight,
        expert_offsets,
    )

    actual = grouped_down(
        hidden,
        down_weight,
        expert_offsets,
    )

    t.cuda.synchronize()

    print(
        "Down max abs error:",
        (actual - expected).abs().max().item(),
    )

    diff = (actual - expected).abs()

    bad_rows = (diff > 0.02).any(dim=1).nonzero().flatten()

    print("expert_offsets:", expert_offsets)
    print("bad rows:", bad_rows)
    print("num bad rows:", len(bad_rows))

    assert actual.shape == expected.shape
    assert t.isfinite(actual).all()

    t.testing.assert_close(
        actual,
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_complete_grouped_expert_forward():
    (
        expert_inputs,
        gate_weight,
        up_weight,
        down_weight,
        expert_offsets,
        *_,
    ) = make_test_data()

    # -------------------------
    # PyTorch reference
    # -------------------------
    reference_hidden = reference_swiglu(
        expert_inputs,
        gate_weight,
        up_weight,
        expert_offsets,
    )

    expected = reference_down(
        reference_hidden,
        down_weight,
        expert_offsets,
    )

    # -------------------------
    # Triton path
    # -------------------------
    triton_hidden = grouped_swiglu_up(
        expert_inputs,
        gate_weight,
        up_weight,
        expert_offsets,
    )

    actual = grouped_down(
        triton_hidden,
        down_weight,
        expert_offsets,
    )

    t.cuda.synchronize()

    print(
        "Full expert path max abs error:",
        (actual - expected).abs().max().item(),
    )

    assert actual.shape == expert_inputs.shape
    assert t.isfinite(actual).all()

    t.testing.assert_close(
        actual,
        expected,
        rtol=3e-2,
        atol=3e-2,
    )