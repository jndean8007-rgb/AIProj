from torch_llm.model.rmsnorm import RMSNorm
import torch as t



def test_norm():
    x = t.randn((3, 4, 5), requires_grad=True)
    rms = RMSNorm(x.size(dim=-1))
    output = rms(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert rms.weight.grad is not None
    assert t.isfinite(x.grad).all()
    assert t.isfinite(rms.weight.grad).all()
