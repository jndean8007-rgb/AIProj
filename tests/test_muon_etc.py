import torch as t

from torch_llm.training.optim.muon import zeropower_via_newton_schulz, Muon


def exact_polar_factor(B: t.Tensor) -> t.Tensor:
    """
    Exact reference using SVD.

    B: (..., M, N)
    returns: (..., M, N)
    """
    U, _, Vh = t.linalg.svd(B.float(), full_matrices=False)
    return U @ Vh


def test_zeropower_matches_svd_square():
    t.manual_seed(0)

    device = "cuda"

    B = t.randn(
        64,
        64,
        device=device,
        dtype=t.float32,
    )

    O_ns = zeropower_via_newton_schulz(
        B,
        steps=5,
    )

    O_exact = exact_polar_factor(B)

    assert O_ns.shape == B.shape
    assert t.isfinite(O_ns).all()

    max_err = (O_ns - O_exact).abs().max().item()
    mean_err = (O_ns - O_exact).abs().mean().item()

    print("square max error:", max_err)
    print("square mean error:", mean_err)

    # NS is approximate, so don't use extremely tight tolerances.
    t.testing.assert_close(
        O_ns,
        O_exact,
        atol=2e-1,
        rtol=2e-1,
    )


def test_zeropower_matches_svd_rectangular():
    t.manual_seed(1)

    device = "cuda"

    # Deliberately tall to test your transpose path.
    B = t.randn(
        128,
        64,
        device=device,
        dtype=t.float32,
    )

    O_ns = zeropower_via_newton_schulz(
        B,
        steps=5,
    )

    O_exact = exact_polar_factor(B)

    assert O_ns.shape == B.shape
    assert t.isfinite(O_ns).all()

    max_err = (O_ns - O_exact).abs().max().item()
    mean_err = (O_ns - O_exact).abs().mean().item()

    print("rectangular max error:", max_err)
    print("rectangular mean error:", mean_err)

    t.testing.assert_close(
        O_ns,
        O_exact,
        atol=2e-1,
        rtol=2e-1,
    )


def test_zeropower_batched_experts():
    t.manual_seed(2)

    device = "cuda"

    # Think of this as E expert matrices.
    E = 4
    F = 128
    D = 64

    B = t.randn(
        E,
        F,
        D,
        device=device,
        dtype=t.float32,
    )

    O_ns = zeropower_via_newton_schulz(
        B,
        steps=5,
    )

    O_exact = exact_polar_factor(B)

    assert O_ns.shape == B.shape
    assert t.isfinite(O_ns).all()

    max_err = (O_ns - O_exact).abs().max().item()
    mean_err = (O_ns - O_exact).abs().mean().item()

    print("batched max error:", max_err)
    print("batched mean error:", mean_err)

    t.testing.assert_close(
        O_ns,
        O_exact,
        atol=2e-1,
        rtol=2e-1,
    )


def test_zeropower_flattens_singular_values():
    t.manual_seed(3)

    device = "cuda"

    B = t.randn(
        96,
        64,
        device=device,
        dtype=t.float32,
    )

    O = zeropower_via_newton_schulz(
        B,
        steps=5,
    )

    singular_values = t.linalg.svdvals(O.float())

    print(
        "singular values:",
        singular_values[:10]
    )

    print(
        "mean singular value:",
        singular_values.mean().item()
    )

    print(
        "std singular value:",
        singular_values.std().item()
    )

    # They should be clustered around ~1,
    # though tuned NS after only 5 steps is approximate.
    assert singular_values.mean() > 0.7
    assert singular_values.mean() < 1.3

def test_muon_step_updates_parameter():
    t.manual_seed(4)

    device = "cuda"

    W = t.nn.Parameter(
        t.randn(
            64,
            64,
            device=device,
            dtype=t.float32,
        )
    )

    optimizer = Muon(
        [W],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        ns_steps=5,
        nesterov=True,
    )

    W_before = W.detach().clone()

    loss = W.square().mean()
    loss.backward()

    optimizer.step()

    assert t.isfinite(W).all()
    assert not t.equal(W, W_before)

    assert "momentum_matrix" in optimizer.state[W]

    momentum = optimizer.state[W]["momentum_matrix"]

    assert momentum.shape == W.shape
    assert t.isfinite(momentum).all()
