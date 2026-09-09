from torch_llm.kernals.flash_attention import FlashAttentionFunction
import torch as t
import math
from einops import rearrange
import itertools


#py -m pytest tests/test_flash_attention.py -v
def test_ragged_flash_attention_forward_backward():
    lengths = [3, 73, 141]

    b = len(lengths)
    hq = 8
    hkv = 2
    d = 16

    T = sum(lengths)
    batch_max_seq_len = max(lengths)
    repeat_factor = hq // hkv
    scale = 1.0 / math.sqrt(d)

    cum_seq = t.tensor(
        [0] + list(itertools.accumulate(lengths)),
        dtype=t.int32,
        device="cuda",
    )

    # Base packed tensors
    q0 = t.randn((T, hq, d), device="cuda", dtype=t.float32)
    k0 = t.randn((T, hkv, d), device="cuda", dtype=t.float32)
    v0 = t.randn((T, hkv, d), device="cuda", dtype=t.float32)

    # Independent autograd paths
    q_ref = q0.detach().clone().requires_grad_(True)
    k_ref = k0.detach().clone().requires_grad_(True)
    v_ref = v0.detach().clone().requires_grad_(True)

    q_tri = q0.detach().clone().contiguous().requires_grad_(True)
    k_tri = k0.detach().clone().contiguous().requires_grad_(True)
    v_tri = v0.detach().clone().contiguous().requires_grad_(True)

    # -------------------------
    # PyTorch ragged reference
    # -------------------------
    reference_parts = []

    for batch_idx in range(b):
        start = int(cum_seq[batch_idx].item())
        end = int(cum_seq[batch_idx + 1].item())
        seq_len = end - start

        q_b = q_ref[start:end]  # (L, Hq, D)
        k_b = k_ref[start:end]  # (L, Hkv, D)
        v_b = v_ref[start:end]

        q_b = rearrange(q_b, "l h d -> h l d")
        k_b = rearrange(k_b, "l h d -> h l d")
        v_b = rearrange(v_b, "l h d -> h l d")

        k_b = k_b.repeat_interleave(repeat_factor, dim=0)
        v_b = v_b.repeat_interleave(repeat_factor, dim=0)

        scores = scale * (q_b @ k_b.transpose(-1, -2))

        causal_mask = t.tril(
            t.ones(
                (seq_len, seq_len),
                dtype=t.bool,
                device="cuda",
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        probs = t.softmax(scores, dim=-1)

        out_b = probs @ v_b                  # (Hq, L, D)
        out_b = rearrange(out_b, "h l d -> l h d")

        reference_parts.append(out_b)

    reference = t.cat(reference_parts, dim=0)

    # -------------------------
    # Triton implementation
    # -------------------------
    actual = FlashAttentionFunction.apply(
        q_tri,
        k_tri,
        v_tri,
        cum_seq,
        batch_max_seq_len,
    )

    # Forward correctness
    t.testing.assert_close(
        actual,
        reference,
        rtol=1e-2,
        atol=1e-2,
    )

    # -------------------------
    # Backward correctness
    # -------------------------
    grad_output = t.randn_like(reference)

    reference.backward(grad_output)
    actual.backward(grad_output)

    t.testing.assert_close(
        q_tri.grad,
        q_ref.grad,
        rtol=1e-2,
        atol=1e-2,
    )

    t.testing.assert_close(
        k_tri.grad,
        k_ref.grad,
        rtol=1e-2,
        atol=1e-2,
    )

    t.testing.assert_close(
        v_tri.grad,
        v_ref.grad,
        rtol=1e-2,
        atol=1e-2,
    )
