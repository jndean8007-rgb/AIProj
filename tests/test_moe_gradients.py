import torch as t

from torch_llm.model.moe.moe import MoE


def test_moe_end_to_end_gradients():
    t.manual_seed(0)

    device = "cuda"
    dtype = t.float16

    T = 13
    D = 64
    F = 128
    E = 4
    K = 2

    moe = MoE(
        num_experts=E,
        top_k=K,
        d_model=D,
        d_ff=F,
        beta=0.9,
        bias_lr = 0.01
    ).to(device=device, dtype=dtype)

    x = t.randn(
        T,
        D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    output, stats = moe(x)

    assert output.shape == (T, D)

    loss = output.float().square().mean()

    # If aux_loss is intended to participate in training:
    loss = loss + 0.01 * stats.aux_loss

    loss.backward()

    # Input gradient
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert t.isfinite(x.grad).all()

    # Router gradient
    router_weight = moe.router.routing_weights.weight

    assert router_weight.grad is not None
    assert t.isfinite(router_weight.grad).all()

    # Expert parameter gradients
    ew = moe.expert_weights

    assert ew.gate_weight.grad is not None
    assert ew.up_weight.grad is not None
    assert ew.down_weight.grad is not None

    assert ew.gate_weight.grad.shape == (E, F, D)
    assert ew.up_weight.grad.shape == (E, F, D)
    assert ew.down_weight.grad.shape == (E, D, F)

    assert t.isfinite(ew.gate_weight.grad).all()
    assert t.isfinite(ew.up_weight.grad).all()
    assert t.isfinite(ew.down_weight.grad).all()