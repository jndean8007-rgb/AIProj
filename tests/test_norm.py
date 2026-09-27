import pytest
import torch as t

from torch_llm.model.rmsnorm import RMSNorm


@pytest.mark.skipif(not t.cuda.is_available(), reason="RMSNorm is a Triton kernel and requires CUDA")
def test_norm():
    # Activations are packed [T, d_model] (handoff D1), so RMSNorm takes 2D input.
    x = t.randn((12, 5), device="cuda", requires_grad=True)
    rms = RMSNorm(x.size(dim=-1)).cuda()
    output = rms(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert rms.weight.grad is not None
    assert t.isfinite(x.grad).all()
    assert t.isfinite(rms.weight.grad).all()
